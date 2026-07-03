import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def read_hdr(path):
    image = iio.imread(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def tonemap(image):
    safe = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.maximum(safe, 0.0)
    mapped = safe / (1.0 + safe)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    return (mapped * 255.0 + 0.5).astype(np.uint8)


def error_image(image, reference):
    diff = np.abs(np.nan_to_num(image - reference, nan=0.0, posinf=0.0, neginf=0.0))
    scalar = np.mean(diff, axis=-1)
    scale = np.percentile(scalar, 99.0)
    if scale <= 1e-8:
        scale = 1.0
    heat = np.clip(scalar / scale, 0.0, 1.0)
    rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * heat).astype(np.uint8)
    rgb[..., 1] = (255.0 * np.maximum(0.0, 1.0 - np.abs(heat - 0.5) * 2.0)).astype(np.uint8)
    rgb[..., 2] = (255.0 * (1.0 - heat)).astype(np.uint8)
    return rgb


def labeled_tile(rgb, label):
    image = Image.fromarray(rgb)
    header_h = 22
    canvas = Image.new("RGB", (image.width, image.height + header_h), (22, 22, 22))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), label, fill=(235, 235, 235))
    return canvas


def write_sheet(items, output_path):
    tiles = [labeled_tile(rgb, label) for label, rgb in items]
    if not tiles:
        return
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    sheet.save(output_path)


def stats_row(name, image, reference):
    finite = np.isfinite(image)
    safe = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    row = {
        "method": name,
        "shape": "x".join(str(v) for v in image.shape),
        "finite": bool(np.all(finite)),
        "nan_count": int(np.isnan(image).sum()),
        "inf_count": int(np.isinf(image).sum()),
        "min": float(np.min(safe)),
        "max": float(np.max(safe)),
        "mean": float(np.mean(safe)),
        "p50": float(np.percentile(safe, 50.0)),
        "p95": float(np.percentile(safe, 95.0)),
        "p99": float(np.percentile(safe, 99.0)),
    }
    if reference is not None and image.shape == reference.shape:
        diff = safe.astype(np.float64) - np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        row["mae"] = float(np.mean(np.abs(diff)))
        row["mse"] = float(np.mean(diff * diff))
    else:
        row["mae"] = ""
        row["mse"] = ""
    return row


def find_method_hdrs(run_dir):
    hdrs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        preferred = child / f"{child.name}.hdr"
        if preferred.exists():
            hdrs.append((child.name, preferred))
            continue
        candidates = sorted(child.glob("*.hdr"))
        if candidates:
            hdrs.append((child.name, candidates[0]))
    return hdrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", default="path_reference")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "hdr_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hdrs = find_method_hdrs(run_dir)
    images = [(name, path, read_hdr(path)) for name, path in hdrs]
    reference = next((image for name, _, image in images if name == args.reference), None)

    write_sheet(
        [(name, tonemap(image)) for name, _, image in images],
        output_dir / "tonemap_sheet.png",
    )
    if reference is not None:
        write_sheet(
            [(name, error_image(image, reference)) for name, _, image in images if image.shape == reference.shape],
            output_dir / "error_sheet.png",
        )

    rows = [stats_row(name, image, reference) for name, _, image in images]
    with (output_dir / "hdr_stats.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)

    print(output_dir)


if __name__ == "__main__":
    main()
