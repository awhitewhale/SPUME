#!/usr/bin/env python3
"""Smoke/train an identity-initialized residual surface-write controller."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def silog_loss(pred, gt):
    valid = torch.isfinite(gt) & (gt > 1e-3) & (gt < 30) & torch.isfinite(pred) & (pred > 1e-6)
    if valid.sum() < 100:
        return None
    delta = torch.log(pred[valid].clamp_min(1e-6)) - torch.log(gt[valid])
    return (delta.square().mean() - 0.85 * delta.mean().square()).clamp_min(0)


def load_gt(paths, height, width, device):
    arrays = [np.asarray(Image.open(path), dtype=np.float32) for path in paths]
    arrays = [cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST) for value in arrays]
    return torch.from_numpy(np.stack(arrays)).to(device)


def gated_depth(model, images, gate, layer, register_residual_write_gate):
    block = model.aggregator.global_blocks[layer]
    handle = register_residual_write_gate(block, gate)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, patch_start = model.aggregator(images[None])
            depth, _ = model.depth_head(tokens, images[None], patch_start, frames_chunk_size=8)
        return depth[0, ..., 0].float()
    finally:
        handle.remove()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(1, str(config["upstream_root"]))

    from datasets.flsea import deterministic_windows, exact_pairs
    from diagnostics.r03_surface_persistence_audit import (
        forward_tokens, inject_transient, load_model, persistence_components, robust_z,
    )
    from models.surface_write_controller import SurfaceWriteController, register_residual_write_gate
    from utils.reproducibility import collect_metadata, set_seed, write_json
    from wat3r.utils.load_fn import load_and_preprocess_images

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(config["upstream_root"], seed, sys.argv)
    metadata["config"] = config
    write_json(output_dir / "metadata.json", metadata)

    model = load_model(config, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = SurfaceWriteController(5, float(config["initial_gate"])).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=float(config["learning_rate"]), weight_decay=0)

    windows = []
    for sequence in config["sequences"]:
        pairs, _ = exact_pairs(config["flsea_root"], sequence)
        selected = deterministic_windows(pairs, int(config["frames_per_window"]), int(config["windows_per_sequence"]))
        windows.extend((sequence, start, window) for start, window in selected)

    rows = []
    for step in range(int(config["steps"])):
        sequence, start, window = windows[step % len(windows)]
        image_paths = [str(item[0]) for item in window]
        depth_paths = [str(item[1]) for item in window]
        images = load_and_preprocess_images(image_paths, mode="max", target_size=518)
        conditions = config.get("conditions", ["clean"])
        condition = str(conditions[step % len(conditions)])
        images, _ = inject_transient(images, condition, seed + 1009 * step)
        with torch.no_grad():
            first = forward_tokens(model, images, device)
            components = persistence_components(
                first, images, [int(x) for x in config["layers"]], int(config["max_correspondence_views"])
            )
            valid = components["valid"]
            evidence = torch.stack([
                robust_z(components["point"], valid),
                robust_z(components["feature"], valid),
                robust_z(components["depth"], valid),
                robust_z(components["normal"], valid),
                robust_z(1 - components["lifetime"], valid),
            ], dim=-1).detach()
            baseline_depth = first["depth"].detach()
        del first, components
        torch.cuda.empty_cache()

        images_gpu = images.to(device)
        gate = controller(evidence)
        prediction = gated_depth(model, images_gpu, gate, int(config["layer"]), register_residual_write_gate)
        gt = load_gt(depth_paths, prediction.shape[-2], prediction.shape[-1], device)
        data_loss = silog_loss(prediction, gt)
        baseline_loss = silog_loss(baseline_depth, gt)
        if data_loss is None or baseline_loss is None:
            continue
        identity_loss = (1 - gate).square().mean()
        loss = data_loss + float(config["identity_weight"]) * identity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), 10).item())
        optimizer.step()
        row = {
            "step": step, "sequence": sequence, "start": start, "condition": condition,
            "loss": float(loss.item()), "data_loss": float(data_loss.item()),
            "baseline_loss": float(baseline_loss.item()), "identity_loss": float(identity_loss.item()),
            "gate_mean": float(gate.mean().item()), "gate_min": float(gate.min().item()),
            "gradient_norm": grad_norm, "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        rows.append(row)
        write_rows(output_dir / "training_metrics.partial.csv", rows)
        torch.save(
            {"controller": controller.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "rows": rows},
            output_dir / "controller.partial.pt",
        )
        print(json.dumps(row), flush=True)
        del prediction, gate, evidence, images_gpu, gt, loss
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No valid optimization step completed")
    write_rows(output_dir / "training_metrics.csv", rows)
    torch.save({"controller": controller.state_dict(), "config": config, "rows": rows}, output_dir / "controller.pt")
    finite = all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]) for row in rows)
    status = {"status": "PASS" if finite and any(row["gradient_norm"] > 0 for row in rows) else "FAIL", "steps": len(rows)}
    write_json(output_dir / "smoke_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
