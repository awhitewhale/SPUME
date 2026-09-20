#!/usr/bin/env python3
"""Fit an interpretable transient-evidence fusion on discovery maps only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["point", "feature", "depth", "normal", "short_lifetime"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--max-patches-per-map", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def matrix(path: Path, rng, cap: int | None):
    data = np.load(path)
    valid = data["valid"].astype(bool) & np.isfinite(data["feature"])
    arrays = [data["point"], data["feature"], data["depth"], data["normal"], 1 - data["lifetime"]]
    x = np.stack([value[valid] for value in arrays], axis=1)
    y = data["injected_mask"][valid].astype(np.uint8)
    finite = np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    if cap and len(y) > cap:
        positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        pos_n = min(len(positives), cap // 2)
        neg_n = min(len(negatives), cap - pos_n)
        chosen = np.concatenate([
            rng.choice(positives, pos_n, replace=False) if pos_n else np.empty(0, int),
            rng.choice(negatives, neg_n, replace=False) if neg_n else np.empty(0, int),
        ])
        x, y = x[chosen], y[chosen]
    return x, y


def condition(path: Path):
    return path.stem.rsplit("_", 1)[-1]


def bootstrap(rows, metric, rng, count):
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence"], []).append(row[metric])
    keys = list(by_sequence)
    samples = []
    for _ in range(count):
        selected = rng.choice(len(keys), len(keys), replace=True)
        sequence_means = []
        for index in selected:
            values = np.asarray(by_sequence[keys[index]], dtype=float)
            sequence_means.append(rng.choice(values, len(values), replace=True).mean())
        samples.append(float(np.mean(sequence_means)))
    center = float(np.mean([np.mean(by_sequence[key]) for key in keys]))
    low, high = np.percentile(samples, [2.5, 97.5])
    return center, float(low), float(high), len(keys)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    discovery = sorted((args.discovery / "maps").glob("*.npz"))
    confirmation = sorted((args.confirmation / "maps").glob("*.npz"))
    train = [matrix(path, rng, args.max_patches_per_map) for path in discovery]
    train = [(x, y) for x, y in train if len(np.unique(y)) == 2]
    x_train = np.concatenate([item[0] for item in train])
    y_train = np.concatenate([item[1] for item in train])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=args.seed),
    )
    model.fit(x_train, y_train)

    rows = []
    for path in confirmation:
        x, y = matrix(path, rng, None)
        if len(np.unique(y)) < 2:
            continue
        score = model.predict_proba(x)[:, 1]
        rows.append({
            "map": path.name,
            "sequence": path.stem.split("_w", 1)[0],
            "condition": condition(path),
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "patches": int(len(y)),
        })
    summary = {"train_maps": len(train), "train_patches": int(len(y_train)), "features": FEATURES, "conditions": {}}
    for name in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == name]
        summary["conditions"][name] = {}
        for metric in ["auroc", "auprc"]:
            mean, low, high, sequences = bootstrap(subset, metric, rng, args.bootstrap)
            summary["conditions"][name][metric] = {
                "mean": mean, "ci95": [low, high], "windows": len(subset), "sequences": sequences,
            }
    scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    summary["standardized_coefficients"] = dict(zip(FEATURES, classifier.coef_[0].tolist()))
    summary["intercept"] = float(classifier.intercept_[0])
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "evidence_fusion.joblib")
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "per_window.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
