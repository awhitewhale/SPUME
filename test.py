#!/usr/bin/env python3
"""Evaluate a trained SPUME depth adapter on held-out cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.depth_output_adapter import DepthOutputAdapter
from train import demo_item, load_cache, resolve_device, set_seed, split_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Directory containing held-out .npz files")
    parser.add_argument("--split", default="test", help="Filename prefix used for cache selection")
    parser.add_argument("--output", type=Path, default=Path("outputs/spume/test_metrics.json"))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--demo", action="store_true", help="Use deterministic generated samples")
    return parser.parse_args()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = (
        np.isfinite(target)
        & (target > 1e-3)
        & (target < 30.0)
        & np.isfinite(prediction)
        & (prediction > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError("An evaluation frame contains no valid depth targets")
    pred = prediction[valid]
    gt = target[valid]
    scale = float(np.sum(pred * gt) / max(float(np.sum(pred * pred)), 1e-9))
    aligned = pred * scale
    ratio = np.maximum(aligned / gt, gt / np.maximum(aligned, 1e-6))
    return {
        "raw_absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "scale_absrel": float(np.mean(np.abs(aligned - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((aligned - gt) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "coverage": float(np.mean(valid)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def main() -> None:
    args = parse_args()
    if not args.demo and args.cache_dir is None:
        raise SystemExit("Provide --cache-dir or use --demo")
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DepthOutputAdapter(
        int(checkpoint.get("hidden", 16)), float(checkpoint.get("max_log_residual", 0.7))
    ).to(device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    if args.demo:
        items = [demo_item(index, args.seed + 1000) for index in range(3)]
        sources = [f"demo_{index:02d}" for index in range(len(items))]
    else:
        paths = sorted(args.cache_dir.glob(f"{args.split}_*.npz"))
        if not paths:
            raise SystemExit(f"No {args.split}_*.npz files found in {args.cache_dir}")
        items = [load_cache(path) for path in paths]
        sources = [path.name for path in paths]

    baseline_rows: list[dict[str, float]] = []
    spume_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for item in items:
            for rgb, depth, confidence, risk, target in split_frames(item):
                rgb_t = torch.from_numpy(rgb[None]).to(device)
                depth_t = torch.from_numpy(depth[None]).to(device)
                confidence_t = torch.from_numpy(confidence[None]).to(device)
                risk_t = torch.from_numpy(risk[None]).to(device)
                prediction, _ = model(rgb_t, depth_t, confidence_t, risk_t)
                baseline_rows.append(metrics(depth, target))
                spume_rows.append(metrics(prediction[0].cpu().numpy(), target))

    result = {
        "device": str(device),
        "frames": len(spume_rows),
        "sources": sources,
        "baseline": mean_metrics(baseline_rows),
        "spume": mean_metrics(spume_rows),
    }
    result["delta_scale_absrel"] = (
        result["spume"]["scale_absrel"] - result["baseline"]["scale_absrel"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
