#!/usr/bin/env python3
"""Frozen Wat3R audit for 3D-addressed surface persistence.

This script does not train a model. It extracts Wat3R predictions and selected
aggregator features, builds camera-compensated correspondence residuals at the
patch grid, evaluates them against injected masks and FLSea metric-depth error,
and optionally performs an identity-friendly second-pass activation gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def add_project_paths(config_path: str):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sys.path.insert(0, str(config["upstream_root"]))
    return project_root, config


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def robust_z(value: torch.Tensor, valid: torch.Tensor):
    x = value[valid & torch.isfinite(value)]
    if x.numel() == 0:
        return torch.zeros_like(value)
    median = x.median()
    mad = (x - median).abs().median().clamp_min(1e-6)
    return ((value - median) / (1.4826 * mad)).clamp(-8, 8)


def sample_map(value: torch.Tensor, grid: torch.Tensor, mode="bilinear"):
    if value.ndim == 2:
        value = value[None]
    return F.grid_sample(
        value[None], grid[None], mode=mode, padding_mode="zeros", align_corners=True
    )[0]


def point_normals(points: torch.Tensor):
    dx = torch.roll(points, -1, dims=2) - torch.roll(points, 1, dims=2)
    dy = torch.roll(points, -1, dims=1) - torch.roll(points, 1, dims=1)
    normals = torch.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1e-6).permute(2, 0, 1)
    normals[:, (0, -1), :] = 0
    normals[:, :, (0, -1)] = 0
    return normals


def resize_tensor(value: torch.Tensor, size, mode="bilinear"):
    if value.ndim == 3:
        value = value[:, None]
    kwargs = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = True
    return F.interpolate(value, **kwargs)


def affine_align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    design = np.stack([p, np.ones_like(p)], axis=1)
    scale, shift = np.linalg.lstsq(design, g, rcond=None)[0]
    return scale * pred + shift


def scale_align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    ratio = np.median(gt[valid] / np.maximum(pred[valid], 1e-6))
    return pred * ratio


def depth_metrics(pred: np.ndarray, gt: np.ndarray):
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 1e-3) & (gt < 30) & (pred > 0)
    if valid.sum() < 100:
        return {k: float("nan") for k in ["raw_absrel", "scale_absrel", "affine_absrel", "rmse", "delta1"]}
    scaled = scale_align(pred, gt, valid)
    aligned = affine_align(pred, gt, valid)
    absrel = lambda x: float(np.mean(np.abs(x[valid] - gt[valid]) / gt[valid]))
    ratio = np.maximum(aligned[valid] / gt[valid], gt[valid] / np.maximum(aligned[valid], 1e-6))
    return {
        "raw_absrel": absrel(pred),
        "scale_absrel": absrel(scaled),
        "affine_absrel": absrel(aligned),
        "rmse": float(np.sqrt(np.mean((aligned[valid] - gt[valid]) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def inject_transient(images: torch.Tensor, condition: str, seed: int):
    if condition == "clean":
        return images.clone(), np.zeros((len(images), *images.shape[-2:]), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    result = images.detach().cpu().numpy().copy()
    count, _, height, width = result.shape
    masks = np.zeros((count, height, width), dtype=np.uint8)
    if condition == "particles":
        bases = [(rng.integers(0, width), rng.integers(0, height), rng.integers(3, 10)) for _ in range(90)]
        velocity = rng.uniform(-10, 10, size=(len(bases), 2))
        for frame in range(count):
            alpha = np.zeros((height, width), dtype=np.float32)
            for index, (x0, y0, radius) in enumerate(bases):
                x = int((x0 + frame * velocity[index, 0]) % width)
                y = int((y0 + frame * velocity[index, 1]) % height)
                cv2.circle(alpha, (x, y), int(radius), float(rng.uniform(0.35, 0.85)), -1)
            masks[frame] = alpha > 0.15
            result[frame] = result[frame] * (1 - alpha[None]) + alpha[None]
    elif condition == "caustics":
        yy, xx = np.mgrid[:height, :width]
        for frame in range(count):
            phase = 0.9 * frame
            wave = np.sin(xx / 19 + yy / 31 + phase) + 0.55 * np.sin(xx / 43 - yy / 17 - 0.7 * phase)
            highlight = np.clip((wave - 0.35) / 1.2, 0, 1).astype(np.float32)
            masks[frame] = highlight > 0.18
            result[frame] = np.clip(result[frame] * (1 + 0.45 * highlight[None]), 0, 1)
    elif condition == "occlusion":
        box_h, box_w = height // 5, width // 5
        for frame in range(count):
            x = int((width * 0.1 + frame * width * 0.08) % (width - box_w))
            y = int(height * 0.35)
            masks[frame, y : y + box_h, x : x + box_w] = 1
            result[frame, :, y : y + box_h, x : x + box_w] = 0.05
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return torch.from_numpy(result).to(images.dtype), masks


def image_flow_controls(images: torch.Tensor, patch_size):
    frames = (images.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    count = len(grays)
    magnitude = np.zeros((count, *patch_size), np.float32)
    fb_error = np.zeros_like(magnitude)
    for index in range(count - 1):
        forward = cv2.calcOpticalFlowFarneback(grays[index], grays[index + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(grays[index + 1], grays[index], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = grays[index].shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        target = np.stack([xx + forward[..., 0], yy + forward[..., 1]], axis=-1)
        back_x = cv2.remap(backward[..., 0], target[..., 0], target[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        back_y = cv2.remap(backward[..., 1], target[..., 0], target[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mag = np.linalg.norm(forward, axis=-1)
        fb = np.sqrt((forward[..., 0] + back_x) ** 2 + (forward[..., 1] + back_y) ** 2)
        mag = cv2.resize(mag, patch_size[::-1], interpolation=cv2.INTER_AREA)
        fb = cv2.resize(fb, patch_size[::-1], interpolation=cv2.INTER_AREA)
        magnitude[index] = np.maximum(magnitude[index], mag)
        magnitude[index + 1] = np.maximum(magnitude[index + 1], mag)
        fb_error[index] = np.maximum(fb_error[index], fb)
        fb_error[index + 1] = np.maximum(fb_error[index + 1], fb)
    return torch.from_numpy(magnitude), torch.from_numpy(fb_error)


def load_model(config, device):
    from wat3r.models.wat3r import Wat3R

    model = Wat3R(enable_track=False, enable_camera=True, enable_point=True, enable_depth=True)
    state = torch.load(config["checkpoint"], map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def forward_tokens(model, images, device, gate=None, gate_layer=11, gate_strength=0.25, gate_mode="input"):
    from wat3r.utils.geometry import unproject_depth_map_to_point_map
    from wat3r.utils.pose_enc import pose_encoding_to_extri_intri

    images = images.to(device)
    handle = None
    if gate is not None:
        gate = gate.to(device)

        def token_multiplier(tokens):
            batch, sequence, patch_h, patch_w = 1, gate.shape[0], gate.shape[1], gate.shape[2]
            token_count = tokens.shape[1] // sequence
            patch_count = patch_h * patch_w
            special = token_count - patch_count
            shaped = tokens.view(batch, sequence, token_count, -1)
            multiplier = torch.ones((sequence, token_count), device=tokens.device, dtype=tokens.dtype)
            multiplier[:, special:] = 1 - gate_strength * (1 - gate.reshape(sequence, -1).to(tokens.dtype))
            return multiplier[None, :, :, None], shaped

        def pre_hook(_module, args, kwargs):
            multiplier, shaped = token_multiplier(args[0])
            changed = (shaped * multiplier).reshape_as(args[0])
            return (changed,) + args[1:], kwargs

        def residual_hook(_module, args, kwargs, output):
            multiplier, shaped_input = token_multiplier(args[0])
            shaped_output = output.view_as(shaped_input)
            changed = shaped_input + multiplier * (shaped_output - shaped_input)
            return changed.reshape_as(output)

        block = model.aggregator.global_blocks[gate_layer]
        if gate_mode == "residual":
            handle = block.register_forward_hook(residual_hook, with_kwargs=True)
        elif gate_mode == "input":
            handle = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
        else:
            raise ValueError(f"Unknown gate mode: {gate_mode}")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float32
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        token_list, patch_start = model.aggregator(images[None])
        pose = model.camera_head(token_list)[-1]
        depth, depth_conf = model.depth_head(token_list, images[None], patch_start, frames_chunk_size=8)
        points, point_conf = model.point_head(token_list, images[None], patch_start, frames_chunk_size=8)
    if handle is not None:
        handle.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, images.shape[-2:])
    depth_np = depth[0, ..., 0].float().cpu().numpy()
    ext_np = extrinsics[0].float().cpu().numpy()
    intr_np = intrinsics[0].float().cpu().numpy()
    depth_points = unproject_depth_map_to_point_map(depth_np[..., None], ext_np, intr_np)
    selected = {layer: token_list[layer][:, :, patch_start:].detach() for layer in [4, 11, 17, 23] if token_list[layer] is not None}
    return {
        "depth": depth[0, ..., 0].float(),
        "depth_conf": depth_conf[0].float(),
        "points": points[0].permute(0, 3, 1, 2).float(),
        "point_conf": point_conf[0].float(),
        "depth_points": torch.from_numpy(depth_points).to(device).permute(0, 3, 1, 2).float(),
        "extrinsics": extrinsics[0].float(),
        "intrinsics": intrinsics[0].float(),
        "features": selected,
        "patch_start": patch_start,
        "elapsed_s": elapsed,
        "peak_memory_mb": peak_mb,
    }


def feature_grids(output, layers, patch_h, patch_w):
    grids = []
    for layer in layers:
        tokens = output["features"][layer][0]
        grid = tokens.reshape(tokens.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2).float()
        grids.append(F.normalize(grid, dim=1, eps=1e-6))
    return grids


def persistence_components(output, images, layers, max_views):
    sequence, _, height, width = output["points"].shape
    patch_h = height // 14
    patch_w = width // 14
    size = (patch_h, patch_w)
    points = resize_tensor(output["points"], size)
    depth = resize_tensor(output["depth"], size)[:, 0]
    confidence = resize_tensor(output["depth_conf"], size)[:, 0]
    normals = torch.stack([point_normals(value) for value in points])
    features = feature_grids(output, layers, patch_h, patch_w)
    components = {name: torch.zeros((sequence, patch_h, patch_w), device=points.device) for name in ["point", "depth", "feature", "normal", "lifetime"]}
    valid_counts = torch.zeros_like(components["point"])
    support_counts = torch.zeros_like(components["point"])
    for ref in range(sequence):
        candidates = [j for j in range(sequence) if j != ref]
        candidates.sort(key=lambda j: abs(j - ref))
        for target in candidates[:max_views]:
            xyz = points[ref].permute(1, 2, 0)
            ext = output["extrinsics"][target]
            intr = output["intrinsics"][target]
            cam = torch.einsum("ij,hwj->hwi", ext[:, :3], xyz) + ext[:, 3]
            z = cam[..., 2]
            u = intr[0, 0] * cam[..., 0] / z.clamp_min(1e-6) + intr[0, 2]
            v = intr[1, 1] * cam[..., 1] / z.clamp_min(1e-6) + intr[1, 2]
            grid = torch.stack([2 * u / (width - 1) - 1, 2 * v / (height - 1) - 1], dim=-1)
            valid = (z > 1e-4) & (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
            sampled_points = sample_map(points[target], grid).permute(1, 2, 0)
            point_residual = torch.linalg.vector_norm(xyz - sampled_points, dim=-1) / torch.linalg.vector_norm(xyz, dim=-1).clamp_min(1e-3)
            sampled_depth = sample_map(depth[target], grid)[0]
            depth_residual = (z - sampled_depth).abs() / sampled_depth.abs().clamp_min(1e-3)
            sampled_normal = sample_map(normals[target], grid)
            normal_residual = 1 - (normals[ref] * sampled_normal).sum(0).abs().clamp(0, 1)
            feature_residuals = []
            for grid_features in features:
                sampled_feature = sample_map(grid_features[target], grid)
                feature_residuals.append(1 - (grid_features[ref] * sampled_feature).sum(0).clamp(-1, 1))
            feature_residual = torch.stack(feature_residuals).mean(0)
            for name, value in [("point", point_residual), ("depth", depth_residual), ("normal", normal_residual), ("feature", feature_residual)]:
                components[name][ref] += torch.where(valid, value, 0)
            valid_counts[ref] += valid
            support_counts[ref] += valid & (point_residual < 0.08) & (depth_residual < 0.08)
    denom = valid_counts.clamp_min(1)
    for name in ["point", "depth", "feature", "normal"]:
        components[name] /= denom
    components["lifetime"] = support_counts / denom
    valid = valid_counts > 0
    risk_logit = (
        0.35 * robust_z(components["point"], valid)
        + 0.25 * robust_z(components["feature"], valid)
        + 0.15 * robust_z(components["depth"], valid)
        + 0.15 * robust_z(components["normal"], valid)
        + 0.10 * robust_z(1 - components["lifetime"], valid)
    )
    components["risk"] = torch.sigmoid(risk_logit)
    flow, fb = image_flow_controls(images, size)
    components["flow"] = flow.to(points.device)
    components["fb_flow"] = fb.to(points.device)
    stacked = torch.stack([grid.mean(1) for grid in features]).mean(0)
    components["feature_2d"] = (stacked - stacked.median(dim=0).values).abs()
    components["confidence_risk"] = -confidence
    components["valid"] = valid
    return components


def patch_gt_and_error(output, depth_paths, patch_size):
    gt_full = [np.asarray(Image.open(path), dtype=np.float32) for path in depth_paths]
    errors, gt_resized = [], []
    pred = output["depth"].detach().cpu().numpy()
    for index, gt in enumerate(gt_full):
        gt_for_pred = cv2.resize(gt, (pred.shape[2], pred.shape[1]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(gt_for_pred) & (gt_for_pred > 1e-3) & (gt_for_pred < 30) & np.isfinite(pred[index])
        aligned = affine_align(pred[index], gt_for_pred, valid)
        err = np.abs(aligned - gt_for_pred) / np.maximum(gt_for_pred, 1e-6)
        # Area interpolation must not mix huge errors from invalid zero-depth pixels
        # into neighboring valid patches. Compute masked patch means explicitly.
        weight = cv2.resize(valid.astype(np.float32), patch_size[::-1], interpolation=cv2.INTER_AREA)
        error_sum = cv2.resize(np.where(valid, err, 0).astype(np.float32), patch_size[::-1], interpolation=cv2.INTER_AREA)
        depth_sum = cv2.resize(np.where(valid, gt_for_pred, 0).astype(np.float32), patch_size[::-1], interpolation=cv2.INTER_AREA)
        errors.append(np.where(weight >= 0.5, error_sum / np.maximum(weight, 1e-6), np.nan))
        gt_resized.append(np.where(weight >= 0.5, depth_sum / np.maximum(weight, 1e-6), np.nan))
    return np.stack(errors), np.stack(gt_resized), gt_full


def ranking_metrics(risk: np.ndarray, label: np.ndarray, continuous_error: np.ndarray, valid: np.ndarray):
    mask = valid & np.isfinite(risk) & np.isfinite(continuous_error)
    y = label[mask].astype(np.uint8)
    score = risk[mask]
    error = continuous_error[mask]
    if len(np.unique(y)) < 2:
        auroc = auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))
    rho = float(spearmanr(score, error).statistic) if len(score) > 2 else float("nan")
    order = np.argsort(score)
    coverages = np.linspace(0.1, 1.0, 10)
    risks = [float(error[order[: max(1, int(len(order) * coverage))]].mean()) for coverage in coverages]
    aurc = float(np.trapz(risks, coverages))
    return auroc, auprc, rho, aurc


def main():
    args = parse_args()
    project_root, config = add_project_paths(args.config)
    from datasets.flsea import deterministic_windows, deterministic_windows_excluding, exact_pairs
    from utils.reproducibility import collect_metadata, set_seed, write_json
    from wat3r.utils.load_fn import load_and_preprocess_images

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(config["upstream_root"], seed, sys.argv)
    metadata["config"] = config
    write_json(output_dir / "metadata.json", metadata)

    excluded_by_sequence = {}
    if config.get("exclude_manifest"):
        prior = json.loads(Path(config["exclude_manifest"]).read_text(encoding="utf-8"))
        prior_windows = prior.get("windows", []) if isinstance(prior, dict) else prior
        for item in prior_windows:
            excluded_by_sequence.setdefault(item["sequence"], []).append(int(item["start"]))

    model = load_model(config, device)
    metrics_rows, geometry_rows, manifest, pairing_audit = [], [], [], []
    for sequence in config["sequences"]:
        pairs, audit = exact_pairs(config["flsea_root"], sequence)
        pairing_audit.append(audit)
        excluded = excluded_by_sequence.get(sequence, [])
        if excluded:
            windows = deterministic_windows_excluding(
                pairs, int(config["frames_per_window"]), int(config["windows_per_sequence"]), excluded
            )
        else:
            windows = deterministic_windows(pairs, int(config["frames_per_window"]), int(config["windows_per_sequence"]))
        for window_index, (start, window) in enumerate(windows):
            image_paths = [str(item[0]) for item in window]
            depth_paths = [str(item[1]) for item in window]
            manifest.append({"sequence": sequence, "window": window_index, "start": start, "images": image_paths, "depths": depth_paths})
            clean_images = load_and_preprocess_images(image_paths, mode="max", target_size=518)
            for condition_index, condition in enumerate(config["conditions"]):
                images, synthetic_mask = inject_transient(clean_images, condition, seed + 1009 * window_index + 97 * condition_index)
                output = forward_tokens(model, images, device)
                height, width = output["depth"].shape[-2:]
                patch_size = (height // 14, width // 14)
                components = persistence_components(
                    output, images, [int(x) for x in config["layers"]], int(config["max_correspondence_views"])
                )
                error, gt_patch, gt_full = patch_gt_and_error(output, depth_paths, patch_size)
                # Correspondence validity alone is insufficient: FLSea depth maps contain
                # invalid/zero pixels. Exclude them before labels, correlations and AURC.
                valid = components["valid"].detach().cpu().numpy()
                valid &= np.isfinite(gt_patch) & (gt_patch > 1e-3) & (gt_patch < 30)
                valid &= np.isfinite(error)
                real_label = np.zeros_like(error, dtype=np.uint8)
                for frame in range(len(error)):
                    values = error[frame][valid[frame]]
                    if len(values):
                        real_label[frame] = error[frame] >= np.quantile(values, 0.75)
                syn_label = np.stack([
                    cv2.resize(mask.astype(np.uint8), patch_size[::-1], interpolation=cv2.INTER_AREA) > 0.2
                    for mask in synthetic_mask
                ])
                risk_maps = {
                    "persistence": components["risk"],
                    "point_innovation": components["point"],
                    "feature_innovation": components["feature"],
                    "depth_support": components["depth"],
                    "local_rigidity": components["normal"],
                    "short_lifetime": 1 - components["lifetime"],
                    "flow": components["flow"],
                    "fb_flow": components["fb_flow"],
                    "feature_2d": components["feature_2d"],
                    "confidence": components["confidence_risk"],
                }
                rng = np.random.default_rng(seed + window_index)
                random_risk = components["risk"].detach().cpu().numpy().copy().reshape(-1)
                rng.shuffle(random_risk)
                risk_maps["random"] = torch.from_numpy(random_risk.reshape(components["risk"].shape)).to(device)
                for method, risk_tensor in risk_maps.items():
                    risk = risk_tensor.detach().cpu().numpy()
                    for label_name, label in [("metric_depth_error_proxy", real_label), ("injected_transient", syn_label)]:
                        auroc, auprc, rho, aurc = ranking_metrics(risk, label, error, valid)
                        metrics_rows.append({
                            "sequence": sequence, "window": window_index, "start": start, "condition": condition,
                            "method": method, "label": label_name, "auroc": auroc, "auprc": auprc,
                            "spearman_error": rho, "aurc": aurc, "valid_patches": int(valid.sum()),
                        })
                baseline_metrics = []
                for frame, gt in enumerate(gt_full):
                    pred = output["depth"][frame].detach().cpu().numpy()
                    gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                    baseline_metrics.append(depth_metrics(pred, gt_resized))
                for key in baseline_metrics[0]:
                    geometry_rows.append({
                        "sequence": sequence, "window": window_index, "start": start, "condition": condition,
                        "intervention": "none", "metric": key,
                        "value": float(np.nanmean([item[key] for item in baseline_metrics])),
                        "latency_s": output["elapsed_s"], "peak_memory_mb": output["peak_memory_mb"],
                    })
                intervention = config.get("intervention", {})
                if intervention.get("enabled", False):
                    for method in intervention.get("methods", ["persistence"]):
                        gate = 1 - risk_maps[method]
                        changed = forward_tokens(
                            model, images, device, gate=gate,
                            gate_layer=int(intervention["layer"]), gate_strength=float(intervention["strength"]),
                            gate_mode=str(intervention.get("mode", "input")),
                        )
                        changed_metrics = []
                        for frame, gt in enumerate(gt_full):
                            pred = changed["depth"][frame].detach().cpu().numpy()
                            gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                            changed_metrics.append(depth_metrics(pred, gt_resized))
                        for key in changed_metrics[0]:
                            geometry_rows.append({
                                "sequence": sequence, "window": window_index, "start": start, "condition": condition,
                                "intervention": method, "metric": key,
                                "value": float(np.nanmean([item[key] for item in changed_metrics])),
                                "latency_s": changed["elapsed_s"], "peak_memory_mb": changed["peak_memory_mb"],
                            })
                map_path = output_dir / "maps" / f"{sequence}_w{window_index:02d}_{condition}.npz"
                map_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    map_path,
                    risk=components["risk"].detach().cpu().numpy().astype(np.float32),
                    point=components["point"].detach().cpu().numpy().astype(np.float32),
                    feature=components["feature"].detach().cpu().numpy().astype(np.float32),
                    depth=components["depth"].detach().cpu().numpy().astype(np.float32),
                    normal=components["normal"].detach().cpu().numpy().astype(np.float32),
                    lifetime=components["lifetime"].detach().cpu().numpy().astype(np.float32),
                    metric_depth_error=error.astype(np.float32),
                    injected_mask=syn_label.astype(np.uint8),
                    valid=valid.astype(np.uint8),
                )
                print(f"AUDIT {sequence} {window_index + 1}/{len(windows)} {condition}", flush=True)
                write_csv(output_dir / "ranking_metrics.partial.csv", metrics_rows)
                write_csv(output_dir / "geometry_metrics.partial.csv", geometry_rows)
    write_csv(output_dir / "ranking_metrics.csv", metrics_rows)
    write_csv(output_dir / "geometry_metrics.csv", geometry_rows)
    write_json(output_dir / "window_manifest.json", {"pairing_audit": pairing_audit, "windows": manifest})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
