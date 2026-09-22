"""Prepare lightweight, reproducible assets for the SPUME project page.

The script never changes the source manuscript or Water3D outputs. It renders
the two method figures, selects representative Water3D frames, and produces
compact WebP assets under docs/static.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

POINT_CLOUDS = {
    "reef-lab": "cv_1426",
    "fish-reef": "cv_575",
    "green-water-rock": "cv_1173",
}


def figure_webp(source: Path, target: Path, max_width: int = 2400) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "WEBP", quality=92, method=6)


def crop_observation_spume_pairs(source: Path, target_dir: Path) -> None:
    """Create aligned observation/SPUME pairs for the interactive reveal."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        columns = {
            "observation": (0.0635, 0.2470),
            "spume": (0.4380, 0.6230),
        }
        rows = {
            "particles": (0.0715, 0.3100),
            "caustics": (0.3235, 0.5605),
            "temporary-occlusion": (0.5730, 0.8105),
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        for label, (y0, y1) in rows.items():
            for kind, (x0, x1) in columns.items():
                panel = image.crop(
                    (round(width * x0), round(height * y0), round(width * x1), round(height * y1))
                )
                panel.save(target_dir / f"{label}-{kind}.webp", "WEBP", quality=94, method=6)


def render_pdf_figure(pdftoppm: Path, pdf: Path, target_png: Path, dpi: int = 180) -> None:
    target_png.parent.mkdir(parents=True, exist_ok=True)
    prefix = target_png.with_suffix("")
    subprocess.run(
        [str(pdftoppm), "-png", "-r", str(dpi), "-singlefile", str(pdf), str(prefix)],
        check=True,
    )


def read_binary_ply(path: Path, max_points: int = 240_000) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline().decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Unsupported PLY format: {path}")
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        count = int(vertex_line.rsplit(" ", 1)[-1])
        dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ]
        )
        vertices = np.fromfile(stream, dtype=dtype, count=count)
    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices = vertices[::step]
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    rgb = np.column_stack((vertices["r"], vertices["g"], vertices["b"]))
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def export_point_cloud(source: Path, target: Path, max_points: int = 60_000) -> None:
    """Export a compact, browser-ready float32 point cloud for WebGL."""
    xyz, rgb = read_binary_ply(source, max_points=max_points * 4)
    center = np.median(xyz, axis=0)
    xyz = xyz - center
    radius = np.linalg.norm(xyz, axis=1)
    keep = radius <= np.quantile(radius, 0.985)
    xyz, rgb = xyz[keep], rgb[keep]

    covariance = np.cov(xyz, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    xyz = xyz @ eigenvectors[:, ::-1]
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz, rgb = xyz[indices], rgb[indices]

    scale = np.quantile(np.linalg.norm(xyz, axis=1), 0.98)
    xyz = np.clip(xyz / max(scale, 1e-6), -1.4, 1.4)
    packed = np.column_stack((xyz, rgb / 255.0)).astype("<f4")
    target.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--water3d-root", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, default=Path("docs"))
    parser.add_argument("--pdftoppm", type=Path, required=True)
    args = parser.parse_args()

    images_dir = args.site_root / "static" / "images"
    figures_dir = images_dir / "figures"
    work_dir = args.site_root.parent / "tmp" / "site-assets"
    work_dir.mkdir(parents=True, exist_ok=True)

    for figure in ("fig1", "fig2"):
        rendered = work_dir / f"{figure}.png"
        render_pdf_figure(args.pdftoppm, args.manuscript_dir / f"{figure}.pdf", rendered)
        figure_webp(rendered, figures_dir / f"{figure}.webp")

    qualitative = work_dir / "qualitative.png"
    render_pdf_figure(
        args.pdftoppm,
        args.manuscript_dir / "qualitative_3d_reconstruction.pdf",
        qualitative,
        dpi=300,
    )
    crop_observation_spume_pairs(qualitative, images_dir / "comparisons")

    for label, scene in POINT_CLOUDS.items():
        export_point_cloud(
            args.water3d_root / scene / "output" / "fused.ply",
            args.site_root / "static" / "point-clouds" / f"{label}.spc",
        )

if __name__ == "__main__":
    main()
