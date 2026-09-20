from pathlib import Path

from datasets.flsea import deterministic_windows, deterministic_windows_excluding, exact_pairs


def test_exact_pairs_rejects_independent_sorting(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "imgs").mkdir(parents=True)
    (scene / "depth").mkdir()
    for name in ["a.tiff", "b.tiff"]:
        (scene / "imgs" / name).touch()
    (scene / "depth" / "a_SeaErra_abs_depth.tif").touch()
    (scene / "depth" / "unrelated_SeaErra_abs_depth.tif").touch()
    pairs, audit = exact_pairs(tmp_path, "scene")
    assert [(a.stem, b.stem) for a, b in pairs] == [("a", "a_SeaErra_abs_depth")]
    assert audit["exact_pairs"] == 1
    assert audit["unmatched_rgb"] == 1
    assert audit["unmatched_depth"] == 1


def test_excluded_windows_share_no_frames():
    pairs = list(range(100))
    discovery = deterministic_windows(pairs, frames_per_window=8, count=4)
    starts = [start for start, _ in discovery]
    confirmation = deterministic_windows_excluding(pairs, 8, 12, starts)
    assert len(confirmation) == 12
    assert all(abs(start - excluded) >= 8 for start, _ in confirmation for excluded in starts)
