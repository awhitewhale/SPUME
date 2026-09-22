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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


SCENES = {
    "reef-lab": ("cv_1426", (1, 100, 200)),
    "fish-reef": ("cv_575", (1, 58, 115)),
    "rock-garden": ("cv_587", (1, 54, 108)),
    "submerged-structure": ("cv_1491", (1, 100, 200)),
    "green-water-rock": ("cv_1173", (1, 5, 10)),
    "shallow-rock": ("cv_1136", (1, 8, 16)),
}


def fit_webp(source: Path, target: Path, size: tuple[int, int], quality: int = 86) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        src_ratio = image.width / image.height
        dst_ratio = size[0] / size[1]
        if src_ratio > dst_ratio:
            crop_width = round(image.height * dst_ratio)
            left = (image.width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, image.height))
        else:
            crop_height = round(image.width / dst_ratio)
            top = (image.height - crop_height) // 2
            image = image.crop((0, top, image.width, top + crop_height))
        image = image.resize(size, Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "WEBP", quality=quality, method=6)


def figure_webp(source: Path, target: Path, max_width: int = 2400) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "WEBP", quality=92, method=6)


def crop_qualitative_panels(source: Path, target_dir: Path) -> None:
    """Extract only the SPUME reconstruction column from the final-paper figure."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        x0, x1 = round(width * 0.438), round(width * 0.623)
        rows = {
            "particles": (0.066, 0.306),
            "caustics": (0.309, 0.548),
            "temporary-occlusion": (0.551, 0.790),
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        for label, (y0, y1) in rows.items():
            panel = image.crop((x0, round(height * y0), x1, round(height * y1)))
            panel.save(target_dir / f"{label}.webp", "WEBP", quality=92, method=6)


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


def render_point_cloud(source: Path, target: Path, size: tuple[int, int] = (1440, 900)) -> None:
    xyz, rgb = read_binary_ply(source)
    center = np.median(xyz, axis=0)
    xyz = xyz - center
    radius = np.linalg.norm(xyz, axis=1)
    keep = radius <= np.quantile(radius, 0.985)
    xyz, rgb = xyz[keep], rgb[keep]

    covariance = np.cov(xyz, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    aligned = xyz @ basis
    # Keep the broadest axis horizontal and the thinnest axis as depth.
    x = aligned[:, 0]
    y = aligned[:, 1]
    depth = aligned[:, 2]
    x_lo, x_hi = np.quantile(x, (0.005, 0.995))
    y_lo, y_hi = np.quantile(y, (0.005, 0.995))
    x = np.clip((x - x_lo) / max(x_hi - x_lo, 1e-6), 0, 1)
    y = np.clip((y - y_lo) / max(y_hi - y_lo, 1e-6), 0, 1)

    width, height = size
    margin = 44
    px = (margin + x * (width - 2 * margin)).astype(np.int32)
    py = (height - margin - y * (height - 2 * margin)).astype(np.int32)
    order = np.argsort(depth)

    canvas = Image.new("RGB", size, "#061821")
    points = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(points, "RGBA")
    for index in order:
        color = tuple(int(channel) for channel in rgb[index])
        draw.ellipse((px[index] - 1, py[index] - 1, px[index] + 1, py[index] + 1), fill=(*color, 210))
    glow = points.filter(ImageFilter.GaussianBlur(1.2))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), glow)
    canvas = Image.alpha_composite(canvas, points).convert("RGB")
    canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "WEBP", quality=88, method=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--water3d-root", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, default=Path("docs"))
    parser.add_argument("--pdftoppm", type=Path, required=True)
    args = parser.parse_args()

    images_dir = args.site_root / "static" / "images"
    figures_dir = images_dir / "figures"
    scenes_dir = images_dir / "scenes"
    clouds_dir = images_dir / "clouds"
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
    crop_qualitative_panels(qualitative, images_dir / "spume-results")

    for label, (scene, frame_numbers) in SCENES.items():
        source_dir = args.water3d_root / scene / "output" / "images"
        for position, frame_number in enumerate(frame_numbers, start=1):
            source = source_dir / f"image_{frame_number:04d}.jpg"
            fit_webp(source, scenes_dir / label / f"frame-{position}.webp", (1280, 720))

    fit_webp(
        args.water3d_root / "cv_1426" / "output" / "images" / "image_0001.jpg",
        images_dir / "hero.webp",
        (1920, 1080),
        quality=90,
    )

    for label, scene in (("reef-lab", "cv_1426"), ("green-water-rock", "cv_1173"), ("shallow-rock", "cv_1136")):
        render_point_cloud(
            args.water3d_root / scene / "output" / "fused.ply",
            clouds_dir / f"{label}.webp",
        )

if __name__ == "__main__":
    main()
