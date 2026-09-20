#!/usr/bin/env python3
"""Train the SPUME risk-conditioned bounded depth adapter."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from models.depth_output_adapter import DepthOutputAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Directory containing train_*.npz files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spume"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--max-log-residual", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--demo", action="store_true", help="Use deterministic generated samples")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ("rgb", "depth", "confidence", "risk", "gt")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {name: np.asarray(data[name], dtype=np.float32) for name in required}


def demo_item(index: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + index)
    frames, height, width = 4, 32, 40
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    base = 2.0 + 0.35 * xx / width + 0.2 * yy / height
    gt = np.stack([base + 0.01 * frame for frame in range(frames)])
    risk = np.zeros_like(gt)
    risk[:, 8:24, 12:30] = 1.0
    bias = (0.08 + 0.015 * rng.standard_normal(gt.shape)).astype(np.float32)
    depth = gt * np.exp(risk * bias)
    rgb = rng.uniform(0.0, 1.0, (frames, 3, height, width)).astype(np.float32)
    confidence = np.clip(1.0 - 0.45 * risk, 0.05, 1.0).astype(np.float32)
    return {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "confidence": confidence,
        "risk": risk.astype(np.float32),
        "gt": gt.astype(np.float32),
    }


def split_frames(item: dict[str, np.ndarray]) -> list[tuple[np.ndarray, ...]]:
    frames = item["depth"].shape[0]
    if item["rgb"].shape[0] != frames:
        raise ValueError("RGB and depth frame counts differ")
    return [
        (item["rgb"][i], item["depth"][i], item["confidence"][i], item["risk"][i], item["gt"][i])
        for i in range(frames)
    ]


def valid_log_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (
        torch.isfinite(target)
        & (target > 1e-3)
        & (target < 30.0)
        & torch.isfinite(prediction)
        & (prediction > 0.0)
    )
    if int(valid.sum()) == 0:
        raise RuntimeError("The batch contains no valid depth targets")
    return (prediction[valid].log() - target[valid].log()).abs().mean()


def batches(samples: list[tuple[np.ndarray, ...]], batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(samples))
    for start in range(0, len(order), batch_size):
        selected = [samples[int(i)] for i in order[start : start + batch_size]]
        yield tuple(np.stack(values) for values in zip(*selected))


def main() -> None:
    args = parse_args()
    if not args.demo and args.cache_dir is None:
        raise SystemExit("Provide --cache-dir or use --demo")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.demo:
        items = [demo_item(index, args.seed) for index in range(6)]
        source_files = [f"demo_{index:02d}" for index in range(len(items))]
    else:
        paths = sorted(args.cache_dir.glob("train_*.npz"))
        if not paths:
            raise SystemExit(f"No train_*.npz files found in {args.cache_dir}")
        items = [load_cache(path) for path in paths]
        source_files = [path.name for path in paths]

    samples = [frame for item in items for frame in split_frames(item)]
    model = DepthOutputAdapter(args.hidden, args.max_log_residual).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    rows: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        for step, batch in enumerate(batches(samples, args.batch_size, rng)):
            rgb, depth, confidence, risk, target = [
                torch.from_numpy(value).to(device) for value in batch
            ]
            prediction, residual = model(rgb, depth, confidence, risk)
            data_loss = valid_log_l1(prediction, target)
            regularizer = residual.abs().mean()
            loss = data_loss + args.residual_weight * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "residual_l1": float(regularizer.detach().cpu()),
                }
            )
        print(json.dumps({"epoch": epoch, "loss": rows[-1]["loss"]}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "max_log_residual": args.max_log_residual,
            "seed": args.seed,
        },
        args.output_dir / "checkpoint.pt",
    )
    with (args.output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "samples": len(samples),
        "sources": source_files,
        "final_loss": rows[-1]["loss"],
        "checkpoint": str((args.output_dir / "checkpoint.pt").resolve()),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
