#!/usr/bin/env python3
"""Aggregate frozen-audit results without treating patches as independent samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--primary-method", default="persistence")
    return parser.parse_args()


def window_bootstrap(frame: pd.DataFrame, value: str, count: int, rng):
    # Hierarchical bootstrap: sequences are primary units; windows are sampled
    # within each selected sequence. Patches and CSV rows are never independent.
    by_sequence = {
        sequence: group.groupby("window", dropna=False)[value].mean().dropna().to_numpy()
        for sequence, group in frame.groupby("sequence", dropna=False)
    }
    by_sequence = {key: values for key, values in by_sequence.items() if len(values)}
    if not by_sequence:
        return np.nan, np.nan, np.nan, 0, 0
    keys = list(by_sequence)
    sequence_means = np.asarray([by_sequence[key].mean() for key in keys])
    samples = []
    for _ in range(count):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        sampled_sequence_means = []
        for index in selected:
            values = by_sequence[keys[index]]
            sampled_sequence_means.append(rng.choice(values, size=len(values), replace=True).mean())
        samples.append(float(np.mean(sampled_sequence_means)))
    center = float(sequence_means.mean())
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    windows = int(sum(len(values) for values in by_sequence.values()))
    return center, float(low), float(high), windows, len(keys)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ranking, geometry = [], []
    for directory in args.audit_dirs:
        rank_path, geo_path = directory / "ranking_metrics.csv", directory / "geometry_metrics.csv"
        if rank_path.exists():
            part = pd.read_csv(rank_path)
            part["audit_dir"] = str(directory)
            ranking.append(part)
        if geo_path.exists():
            part = pd.read_csv(geo_path)
            part["audit_dir"] = str(directory)
            geometry.append(part)
    args.output.mkdir(parents=True, exist_ok=True)

    rank = pd.concat(ranking, ignore_index=True) if ranking else pd.DataFrame()
    geo = pd.concat(geometry, ignore_index=True) if geometry else pd.DataFrame()
    rank_rows = []
    if not rank.empty:
        group_cols = ["condition", "method", "label"]
        for keys, group in rank.groupby(group_cols, dropna=False):
            for value in ["auroc", "auprc", "spearman_error", "aurc"]:
                center, low, high, windows, sequences = window_bootstrap(group, value, args.bootstrap, rng)
                rank_rows.append(dict(zip(group_cols, keys)) | {
                    "metric": value, "mean": center, "ci95_low": low, "ci95_high": high,
                    "windows": windows, "sequences": sequences,
                })
    rank_summary = pd.DataFrame(rank_rows)
    rank_summary.to_csv(args.output / "ranking_summary.csv", index=False)

    geo_rows = []
    if not geo.empty:
        group_cols = ["condition", "intervention", "metric"]
        for keys, group in geo.groupby(group_cols, dropna=False):
            center, low, high, windows, sequences = window_bootstrap(group, "value", args.bootstrap, rng)
            geo_rows.append(dict(zip(group_cols, keys)) | {
                "mean": center, "ci95_low": low, "ci95_high": high,
                "windows": windows, "sequences": sequences,
            })
    pd.DataFrame(geo_rows).to_csv(args.output / "geometry_summary.csv", index=False)

    decision = {"status": "INSUFFICIENT", "reasons": []}
    if not rank_summary.empty:
        target = rank_summary[
            (rank_summary.label == "injected_transient")
            & (rank_summary.method == args.primary_method)
            & (rank_summary.metric == "auroc")
        ]
        if len(target) >= 2 and target.windows.min() >= 4:
            mean_auroc = float(target["mean"].mean())
            decision = {
                "status": "GO" if mean_auroc >= 0.70 else "NO_GO",
                "criterion": f"mean {args.primary_method} AUROC across synthetic conditions >= 0.70",
                "observed": mean_auroc,
                "conditions": target[["condition", "mean", "ci95_low", "ci95_high", "windows"]].to_dict("records"),
                "reasons": [],
            }
    (args.output / "gate_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
