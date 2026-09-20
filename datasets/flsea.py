from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairAudit:
    sequence: str
    rgb_count: int
    depth_count: int
    exact_pairs: int
    unmatched_rgb: int
    unmatched_depth: int


def exact_pairs(root: str | Path, sequence: str):
    scene = Path(root) / sequence
    images = sorted((scene / "imgs").glob("*.tiff"))
    depths = sorted((scene / "depth").glob("*.tif"))
    pairs = []
    matched_depths = set()
    for image in images:
        depth = scene / "depth" / f"{image.stem}_SeaErra_abs_depth.tif"
        if depth.exists():
            pairs.append((image, depth))
            matched_depths.add(depth.resolve())
    audit = PairAudit(
        sequence=sequence,
        rgb_count=len(images),
        depth_count=len(depths),
        exact_pairs=len(pairs),
        unmatched_rgb=len(images) - len(pairs),
        unmatched_depth=len(depths) - len(matched_depths),
    )
    return pairs, asdict(audit)


def deterministic_windows(pairs, frames_per_window: int, count: int):
    if len(pairs) < frames_per_window:
        return []
    possible = len(pairs) - frames_per_window + 1
    if count >= possible:
        starts = list(range(possible))
    else:
        starts = np.linspace(0, possible - 1, count).round().astype(int).tolist()
    return [(start, pairs[start : start + frames_per_window]) for start in starts]


def deterministic_windows_excluding(pairs, frames_per_window: int, count: int, excluded_starts):
    """Select evenly spaced windows without frame overlap with discovery starts."""
    if len(pairs) < frames_per_window:
        return []
    possible = len(pairs) - frames_per_window + 1
    excluded_starts = [int(value) for value in excluded_starts]
    allowed = [
        start for start in range(possible)
        if all(abs(start - excluded) >= frames_per_window for excluded in excluded_starts)
    ]
    if not allowed:
        return []
    if count >= len(allowed):
        starts = allowed
    else:
        indices = np.linspace(0, len(allowed) - 1, count).round().astype(int)
        starts = [allowed[index] for index in indices]
    return [(start, pairs[start : start + frames_per_window]) for start in starts]
