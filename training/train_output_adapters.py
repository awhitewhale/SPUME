#!/usr/bin/env python3
"""Train and development-evaluate risk/no-risk/random matched depth adapters."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml


def load_item(path, device):
    data = np.load(path)
    return {key: torch.from_numpy(data[key].astype(np.float32)).to(device) for key in ["rgb", "depth", "confidence", "risk", "gt"]}


def context_for(method, risk, name, seed):
    if method == "risk":
        return risk
    if method == "zero":
        return torch.zeros_like(risk)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + zlib.crc32(name.encode("utf-8")))
    order = torch.randperm(risk.numel(), generator=generator).to(risk.device)
    return risk.reshape(-1)[order].reshape_as(risk)


def log_l1(prediction, gt):
    valid = torch.isfinite(gt) & (gt > 1e-3) & (gt < 30) & torch.isfinite(prediction) & (prediction > 0)
    return (prediction[valid].log() - gt[valid].log()).abs().mean()


def depth_metrics(prediction, gt):
    valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 30) & np.isfinite(prediction) & (prediction > 0)
    if valid.sum() < 100:
        return {"raw_absrel": np.nan, "scale_absrel": np.nan, "delta1": np.nan, "coverage": 0.0}
    scale = np.sum(prediction[valid] * gt[valid]) / max(np.sum(prediction[valid] ** 2), 1e-9)
    scaled = prediction * scale
    raw = np.mean(np.abs(prediction[valid] - gt[valid]) / gt[valid])
    rel = np.mean(np.abs(scaled[valid] - gt[valid]) / gt[valid])
    ratio = np.maximum(scaled[valid] / gt[valid], gt[valid] / np.maximum(scaled[valid], 1e-6))
    return {"raw_absrel": float(raw), "scale_absrel": float(rel), "delta1": float(np.mean(ratio < 1.25)), "coverage": float(valid.mean())}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[1]
    sys.path.insert(0, str(project_root))
    from models.depth_output_adapter import DepthOutputAdapter
    from utils.reproducibility import set_seed, write_json

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    cache_dir, output_dir = Path(config["cache_dir"]), Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(cache_dir.glob("train_*.npz"))
    dev_paths = sorted(cache_dir.glob("development_*.npz"))
    evaluation_split = str(config.get("evaluation_split", "development"))
    if not train_paths or (evaluation_split == "development" and not dev_paths):
        raise RuntimeError("Adapter cache is incomplete")
    eval_paths = train_paths if evaluation_split == "train" else dev_paths

    template = DepthOutputAdapter(int(config["hidden"]), float(config["max_log_residual"])).to(device)
    initial = copy.deepcopy(template.state_dict())
    methods = ["risk", "random", "zero"]
    models, optimizers = {}, {}
    for method in methods:
        model = DepthOutputAdapter(int(config["hidden"]), float(config["max_log_residual"])).to(device)
        model.load_state_dict(initial)
        models[method] = model
        optimizers[method] = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
        )

    train_rows = []
    skipped_training_items = 0
    for epoch in range(int(config["epochs"])):
        order = np.random.default_rng(seed + epoch).permutation(len(train_paths))
        for item_index in order:
            path = train_paths[int(item_index)]
            item = load_item(path, device)
            for method in methods:
                context = context_for(method, item["risk"], path.name, seed)
                prediction, residual = models[method](item["rgb"], item["depth"], item["confidence"], context)
                data_loss = log_l1(prediction, item["gt"])
                if not torch.isfinite(data_loss):
                    skipped_training_items += 1
                    continue
                regularizer = residual.abs().mean()
                loss = data_loss + 0.001 * regularizer
                optimizers[method].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[method].step()
                train_rows.append({
                    "epoch": epoch, "item": path.name, "method": method,
                    "loss": float(loss.item()), "data_loss": float(data_loss.item()),
                    "residual_l1": float(regularizer.item()),
                })
            if len(train_rows) % 24 == 0:
                write_csv(output_dir / "training_metrics.partial.csv", train_rows)
        for method in methods:
            torch.save(models[method].state_dict(), output_dir / f"adapter_{method}.partial.pt")
        recent = train_rows[-len(train_paths) * 3:]
        print(json.dumps({"epoch": epoch, "mean_recent_loss": float(np.mean([row["loss"] for row in recent]))}), flush=True)

    write_csv(output_dir / "training_metrics.csv", train_rows)
    for method in methods:
        torch.save(models[method].state_dict(), output_dir / f"adapter_{method}.pt")
        models[method].eval()

    rows = []
    with torch.no_grad():
        for path in eval_paths:
            item = load_item(path, device)
            cached = np.load(path)
            sequence = str(cached["sequence"].item())
            condition = str(cached["condition"].item())
            window_index = int(cached["window_index"].item())
            predictions = {"wat3r": item["depth"]}
            for method in methods:
                context = context_for(method, item["risk"], path.name, seed)
                predictions[method] = models[method](item["rgb"], item["depth"], item["confidence"], context)[0]
            gt = item["gt"].cpu().numpy()
            for method, prediction in predictions.items():
                pred = prediction.cpu().numpy()
                per_frame = [depth_metrics(pred[index], gt[index]) for index in range(len(gt))]
                for metric in per_frame[0]:
                    rows.append({
                        "sequence": sequence, "window": window_index, "condition": condition,
                        "method": method, "metric": metric,
                        "value": float(np.nanmean([frame[metric] for frame in per_frame])),
                    })
    metrics_name = "in_sample_metrics.csv" if evaluation_split == "train" else "development_metrics.csv"
    write_csv(output_dir / metrics_name, rows)

    import pandas as pd
    frame = pd.DataFrame(rows)
    scale = frame[frame.metric == "scale_absrel"].pivot_table(
        index=["sequence", "window", "condition"], columns="method", values="value"
    ).reset_index()
    corrupt = scale[scale.condition != "clean"]
    clean = scale[scale.condition == "clean"]
    means = corrupt[["wat3r", "risk", "random", "zero"]].mean()
    improvements = {
        "vs_no_risk": float((means["zero"] - means["risk"]) / means["zero"]),
        "vs_random": float((means["random"] - means["risk"]) / means["random"]),
        "vs_wat3r": float((means["wat3r"] - means["risk"]) / means["wat3r"]),
    }
    sequence_improvement = {
        sequence: float((group.wat3r.mean() - group.risk.mean()) / group.wat3r.mean())
        for sequence, group in corrupt.groupby("sequence")
    }
    clean_means = clean[["wat3r", "risk"]].mean()
    clean_degradation = float((clean_means["risk"] - clean_means["wat3r"]) / clean_means["wat3r"])
    coverage = frame[frame.metric == "coverage"].pivot_table(
        index=["sequence", "window", "condition"], columns="method", values="value"
    )
    coverage_loss = float((coverage["wat3r"] - coverage["risk"]).mean())
    gate = config["success_gate"]
    checks = {
        "corrupt_vs_no_risk": improvements["vs_no_risk"] >= float(gate["corrupt_vs_no_risk"]),
        "corrupt_vs_random": improvements["vs_random"] >= float(gate["corrupt_vs_random"]),
        "corrupt_vs_wat3r": improvements["vs_wat3r"] >= float(gate["corrupt_vs_wat3r"]),
        "both_sequences": all(value > 0 for value in sequence_improvement.values()) and len(sequence_improvement) == 2,
        "clean_preserved": clean_degradation <= float(gate["clean_max_degradation"]),
        "coverage_preserved": coverage_loss <= float(gate["max_coverage_loss"]),
    }
    decision = {
        "status": "EXPLORATORY_DATA_LEAKAGE" if evaluation_split == "train" else ("GO" if all(checks.values()) else "DISCARD"),
        "evaluation_split": evaluation_split,
        "training_files": len(train_paths),
        "evaluation_files": len(eval_paths),
        "skipped_training_items": skipped_training_items,
        "checks": checks, "improvements": improvements,
        "sequence_improvement_vs_wat3r": sequence_improvement,
        "clean_degradation": clean_degradation, "coverage_loss": coverage_loss,
        "corrupt_scale_absrel_means": means.to_dict(),
    }
    decision_name = "in_sample_decision.json" if evaluation_split == "train" else "development_decision.json"
    write_json(output_dir / decision_name, decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
