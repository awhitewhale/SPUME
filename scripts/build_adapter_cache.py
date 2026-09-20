#!/usr/bin/env python3
"""Build a frozen Wat3R cache for matched output-adapter experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(1, str(config["upstream_root"]))

    from datasets.flsea import deterministic_windows, exact_pairs
    from diagnostics.r03_surface_persistence_audit import (
        forward_tokens, inject_transient, load_model, persistence_components, robust_z,
    )
    from utils.reproducibility import collect_metadata, set_seed, write_json
    from wat3r.utils.load_fn import load_and_preprocess_images

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(config["upstream_root"], seed, sys.argv)
    metadata["config"] = config
    write_json(cache_dir / "metadata.json", metadata)
    model = load_model(config, device)
    manifest = []
    for sequence in config["sequences"]:
        pairs, audit = exact_pairs(config["flsea_root"], sequence)
        windows = deterministic_windows(
            pairs, int(config["frames_per_window"]), int(config["windows_per_sequence"])
        )
        for window_index, (start, window) in enumerate(windows):
            split = "train" if window_index < int(config["train_windows_per_sequence"]) else "development"
            image_paths = [str(item[0]) for item in window]
            depth_paths = [str(item[1]) for item in window]
            clean = load_and_preprocess_images(image_paths, mode="max", target_size=518)
            for condition_index, condition in enumerate(config["conditions"]):
                images, injected = inject_transient(
                    clean, condition, seed + 100003 * window_index + 1009 * condition_index
                )
                output = forward_tokens(model, images, device)
                height, width = output["depth"].shape[-2:]
                components = persistence_components(
                    output, images, [int(x) for x in config["layers"]], int(config["max_correspondence_views"])
                )
                feature_risk = torch.sigmoid(robust_z(components["feature"], components["valid"]))
                feature_risk = F.interpolate(
                    feature_risk[:, None], (height, width), mode="bilinear", align_corners=False
                )[:, 0]
                gt = np.stack([
                    cv2.resize(np.asarray(Image.open(path), dtype=np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
                    for path in depth_paths
                ])
                name = f"{split}_{sequence}_w{window_index:02d}_{condition}.npz"
                np.savez_compressed(
                    cache_dir / name,
                    sequence=np.asarray(sequence),
                    condition=np.asarray(condition),
                    window_index=np.asarray(window_index, dtype=np.int32),
                    start=np.asarray(start, dtype=np.int32),
                    rgb=images.numpy().astype(np.float16),
                    depth=output["depth"].cpu().numpy().astype(np.float32),
                    confidence=output["depth_conf"].cpu().numpy().astype(np.float16),
                    risk=feature_risk.cpu().numpy().astype(np.float16),
                    gt=gt.astype(np.float32),
                    injected_mask=injected.astype(np.uint8),
                )
                manifest.append({
                    "file": name, "sequence": sequence, "window": window_index, "start": start,
                    "split": split, "condition": condition, "pairing_audit": audit,
                })
                print(json.dumps(manifest[-1]), flush=True)
                write_json(cache_dir / "manifest.partial.json", manifest)
                del output, components, feature_risk
                torch.cuda.empty_cache()
    write_json(cache_dir / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "items": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
