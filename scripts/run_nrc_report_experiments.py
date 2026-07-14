import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "target" / "bin" / "Moer_r.exe"
SCENES = REPO / "scenes"
RESULTS = REPO / "results" / "nrc_report_final"


def nrc_base():
    return {
        "mode": "nrc",
        "query_bounce": 2,
        "train_steps": 0,
        "train_batch_size": 512,
        "training_spp": 24,
        "final_train_steps": 4096,
        "final_train_epochs": 32.0,
        "freeze_after_training": True,
        "cache_non_diffuse_surfaces": True,
        "target_samples": 4,
        "tcnn_hidden_layers": 2,
        "tcnn_relative_target": True,
        "tcnn_weighted_sample_training": True,
        "tcnn_training_weight_clamp": 8.0,
        "use_batch_query": True,
        "min_training_samples_before_query": 1024,
        "max_training_samples": 524288,
        "count_zero_target_samples": True,
        "target_luminance_clamp": 8.0,
    }


def scaled_nrc_for_resolution(resolution):
    settings = nrc_base()
    base_pixels = 1024 * 576
    pixels = resolution[0] * resolution[1]
    scaled = int(math.ceil(524288 * pixels / base_pixels))
    # Round up to a power of two to keep the reservoir capacity predictable.
    settings["max_training_samples"] = 1 << (scaled - 1).bit_length()
    return settings


REPORT_SCENES = [
    ("testball", "scenes/testball", (1920, 1080)),
    ("teapot", "scenes/teapot", (1920, 1080)),
    ("disney", "scenes/disney-bsdf", (1920, 1080)),
    ("classroom", "scenes/classroom", (1920, 1080)),
    ("green_bathroom", "scenes/green-bathroom", (1920, 1080)),
]


SCENE_DISPLAY_EXPOSURE = {
    "classroom": 2.0,
    "green_bathroom": 1.5,
}


def report_exposure(scene_name):
    return SCENE_DISPLAY_EXPOSURE.get(scene_name, 1.0)


def apply_report_scene_adjustments(scene_name, scene):
    if scene_name == "green_bathroom":
        camera = scene.setdefault("camera", {})
        camera["type"] = "pinhole"
        camera.pop("aperture_radius", None)
        camera.pop("focus_distance", None)
    return scene


def methods_for_report(reference_spp=256, include_variants=True, resolution=(1024, 576)):
    best = scaled_nrc_for_resolution(resolution)
    methods = [
        ("path_low", 4, {"mode": "path_trace", "query_bounce": 1}),
        ("path_visual", 32, {"mode": "path_trace", "query_bounce": 1}),
        ("path_equal", 128, {"mode": "path_trace", "query_bounce": 1}),
        ("nrc_best", 128, best),
    ]
    methods.append(("path_reference", reference_spp, {"mode": "path_trace", "query_bounce": 1}))
    return methods


def ablation_methods():
    base = {
        "mode": "nrc",
        "query_bounce": 2,
        "train_steps": 0,
        "train_batch_size": 512,
        "training_spp": 8,
        "final_train_steps": 512,
        "freeze_after_training": True,
        "cache_non_diffuse_surfaces": True,
        "target_samples": 1,
        "tcnn_hidden_layers": 2,
        "tcnn_relative_target": False,
        "tcnn_weighted_sample_training": False,
        "use_batch_query": False,
        "min_training_samples_before_query": 1024,
        "max_training_samples": 65536,
        "count_zero_target_samples": False,
        "target_luminance_clamp": 0.0,
    }
    batch = dict(base, use_batch_query=True)
    avg = dict(batch, target_samples=4)
    rel = dict(avg, tcnn_relative_target=True)
    weighted = dict(rel, tcnn_weighted_sample_training=True, tcnn_training_weight_clamp=8.0)
    zero = dict(weighted, count_zero_target_samples=True)
    clamp = dict(zero, target_luminance_clamp=8.0)
    more = dict(clamp,
                training_spp=24,
                final_train_steps=4096,
                final_train_epochs=32.0,
                target_samples=8,
                max_training_samples=524288)
    return [
        ("path_low", 2, {"mode": "path_trace", "query_bounce": 1}),
        ("path_reference", 32, {"mode": "path_trace", "query_bounce": 1}),
        ("nrc_00_basic", 2, base),
        ("nrc_01_batch_query", 2, batch),
        ("nrc_02_target_avg4", 2, avg),
        ("nrc_03_relative_loss", 2, rel),
        ("nrc_04_weighted", 2, weighted),
        ("nrc_05_zero_targets", 2, zero),
        ("nrc_06_lum_clamp", 2, clamp),
        ("nrc_07_best_training", 48, more),
    ]


def read_scene(scene_dir):
    with open(scene_dir / "scene.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def link_or_copy_asset(src, dst):
    if dst.exists():
        return
    if src.is_dir():
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst)
    elif src.is_file():
        shutil.copy2(src, dst)


def prepare_scene(source_scene, scene_name, method_name, spp, nrc, resolution, run_root):
    source_dir = REPO / source_scene
    method_dir = run_root / "raw" / scene_name / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name != "scene.json":
            link_or_copy_asset(item, method_dir / item.name)

    scene = read_scene(source_dir)
    if isinstance(scene.get("camera", {}).get("resolution"), list):
        scene["camera"]["resolution"] = [resolution[0], resolution[1]]
    else:
        scene.setdefault("camera", {})["resolution"] = [resolution[0], resolution[1]]
    apply_report_scene_adjustments(scene_name, scene)
    renderer = scene.setdefault("renderer", {})
    renderer["spp"] = spp
    renderer["integrator"] = "nrc_path"
    renderer["nrc"] = nrc
    renderer["overwrite_output_files"] = True
    renderer["enable_resume_render"] = False
    renderer["output_file"] = str(method_dir / method_name)
    renderer["hdr_output_file"] = method_name + ".hdr"
    write_json(method_dir / "scene.json", scene)
    return method_dir


def run_render(method_dir, timeout):
    started = time.perf_counter()
    proc = subprocess.run(
        [str(EXE), str(method_dir)],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    seconds = time.perf_counter() - started
    (method_dir / "render.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"render failed: {method_dir} code={proc.returncode}")
    return seconds


def hdr_path(method_dir, method_name):
    direct = method_dir / f"{method_name}.hdr"
    if direct.exists():
        return direct
    matches = sorted(method_dir.glob(f"{method_name}*.hdr"))
    if not matches:
        raise FileNotFoundError(f"missing hdr for {method_dir}")
    return matches[-1]


def read_hdr(path):
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return img[:, :, :3]


def report_display_map(img, exposure=1.0):
    img = np.maximum(img, 0.0)
    img = img * exposure
    if float(np.nanmax(img)) > 8.0:
        return np.clip(img / 255.0, 0.0, 1.0)
    mapped = img / (1.0 + img)
    return np.clip(np.power(mapped, 1.0 / 2.2), 0.0, 1.0)


def save_png(hdr, png_path, exposure=1.0):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    ldr = (report_display_map(hdr, exposure) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(ldr).save(png_path)


def metrics(image, reference, exposure=1.0):
    if image.shape != reference.shape:
        image = cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    diff = image - reference
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    ref_mean = float(np.mean(reference))
    rel_mae = float(np.mean(np.abs(diff) / (np.abs(reference) + 1.0)))
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10((max(1.0, float(np.max(reference))) ** 2) / mse)

    img_ldr = report_display_map(image, exposure)
    ref_ldr = report_display_map(reference, exposure)
    ldr_mse = float(np.mean((img_ldr - ref_ldr) ** 2))
    ldr_psnr = float("inf") if ldr_mse <= 0 else 10.0 * math.log10(1.0 / ldr_mse)
    ldr_mae = float(np.mean(np.abs(img_ldr - ref_ldr)))
    return {
        "mse": mse,
        "mae": mae,
        "relative_mae": rel_mae,
        "psnr": psnr,
        "reference_mean": ref_mean,
        "ldr_mae": ldr_mae,
        "ldr_psnr": ldr_psnr,
    }


def read_csv_key_values(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    if rows[0] == ["metric", "value"]:
        return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}
    header = rows[0]
    values = rows[1] if len(rows) > 1 else []
    return {k: values[i] if i < len(values) else "" for i, k in enumerate(header)}


def add_text_label(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)


def make_comparison(scene_name, labels, png_paths, out_path):
    images = [Image.open(p).convert("RGB") for p in png_paths]
    min_h = min(im.height for im in images)
    resized = [im.resize((int(im.width * min_h / im.height), min_h), Image.Resampling.LANCZOS) for im in images]
    for im, label in zip(resized, labels):
        add_text_label(im, label)
    total_w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (total_w, min_h), (0, 0, 0))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_table(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_suite(suite, run_root, timeout):
    rows = []
    png_by_scene = {}
    timings = {}
    for scene_name, scene_path, resolution, methods in suite:
        refs = {}
        png_by_scene[scene_name] = {}
        for method_name, spp, nrc in methods:
            method_dir = prepare_scene(scene_path, scene_name, method_name, spp, nrc, resolution, run_root)
            print(f"[render] {scene_name}/{method_name} spp={spp} res={resolution[0]}x{resolution[1]}", flush=True)
            seconds = run_render(method_dir, timeout)
            timings[(scene_name, method_name)] = seconds
            hdr = hdr_path(method_dir, method_name)
            image = read_hdr(hdr)
            png = run_root / "png" / scene_name / f"{method_name}.png"
            save_png(image, png, report_exposure(scene_name))
            png_by_scene[scene_name][method_name] = png
            if method_name == "path_reference":
                refs[scene_name] = image
        reference = refs.get(scene_name)
        if reference is None:
            ref_hdr = hdr_path(run_root / "raw" / scene_name / "path_reference", "path_reference")
            reference = read_hdr(ref_hdr)

        for method_name, spp, _ in methods:
            method_dir = run_root / "raw" / scene_name / method_name
            image = read_hdr(hdr_path(method_dir, method_name))
            row = {
                "scene": scene_name,
                "method": method_name,
                "spp": spp,
                "resolution": f"{resolution[0]}x{resolution[1]}",
            }
            row.update(metrics(image, reference, report_exposure(scene_name)))
            row["render_seconds"] = timings.get((scene_name, method_name), "")
            row.update({f"stats_{k}": v for k, v in read_csv_key_values(method_dir / f"{method_name}.stats.csv").items()})
            row.update({f"cache_{k}": v for k, v in read_csv_key_values(method_dir / f"{method_name}.stats.csv.cache.csv").items()})
            rows.append(row)

        ordered = [m[0] for m in methods if m[0] in png_by_scene[scene_name]]
        important = [m for m in ["path_low", "path_visual", "path_equal", "nrc_best", "path_reference"] if m in ordered]
        if len(important) >= 2:
            make_comparison(
                scene_name,
                important,
                [png_by_scene[scene_name][m] for m in important],
                run_root / "comparisons" / f"{scene_name}_comparison.png",
            )
        if {"path_equal", "nrc_best", "path_reference"}.issubset(png_by_scene[scene_name]):
            make_comparison(
                scene_name,
                ["path_equal", "nrc_best", "path_reference"],
                [png_by_scene[scene_name][m] for m in ["path_equal", "nrc_best", "path_reference"]],
                run_root / "comparisons" / f"{scene_name}_triple.png",
            )
        if {"path_visual", "nrc_best", "path_reference"}.issubset(png_by_scene[scene_name]):
            make_comparison(
                scene_name,
                ["path_visual", "nrc_best", "path_reference"],
                [png_by_scene[scene_name][m] for m in ["path_visual", "nrc_best", "path_reference"]],
                run_root / "comparisons" / f"{scene_name}_visual_triple.png",
            )
    return rows


def summarize(rows, run_root):
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = [
        "scene", "method", "spp", "resolution", "render_seconds", "mse", "mae", "relative_mae",
        "psnr", "ldr_mae", "ldr_psnr", "stats_query_count", "stats_training_samples",
        "stats_train_calls", "stats_nrc_query_seconds", "stats_nrc_train_seconds",
        "stats_target_trace_seconds", "cache_final_loss", "cache_validation_decoded_mae",
        "cache_validation_decoded_rel_mae", "cache_observed_sample_count",
    ]
    ordered = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    write_table(run_root / "tables" / "metrics.csv", rows, ordered)

    lines = ["# NRC Report Results", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Commit: {git_commit()}")
    lines.append("")
    lines.append("## Files")
    lines.append("- `raw/`: HDR outputs, scene JSONs, stats CSVs, render logs")
    lines.append("- `png/`: tone-mapped PNGs")
    lines.append("- `comparisons/`: side-by-side report figures")
    lines.append("- `tables/metrics.csv`: quantitative metrics against `path_reference`")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("| scene | method | spp | LDR PSNR | LDR MAE | linear MAE | seconds |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.get('scene','')} | {row.get('method','')} | {row.get('spp','')} | "
            f"{fmt(row.get('ldr_psnr'))} | {fmt(row.get('ldr_mae'))} | {fmt(row.get('mae'))} | {row.get('render_seconds','')} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.4g}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "ablation", "report", "report_fast", "report_720"], default="smoke")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--scene-names", nargs="*", default=[])
    args = parser.parse_args()

    if not EXE.exists():
        raise FileNotFoundError(EXE)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = RESULTS / f"{args.suite}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.suite == "smoke":
        smoke_methods = [
            ("path_low", 1, {"mode": "path_trace", "query_bounce": 1}),
            ("nrc_best", 1, dict(nrc_base(),
                                  query_bounce=1,
                                  training_spp=1,
                                  final_train_steps=1,
                                  final_train_epochs=0.0,
                                  target_samples=1,
                                  min_training_samples_before_query=1,
                                  max_training_samples=512)),
            ("path_reference", 1, {"mode": "path_trace", "query_bounce": 1}),
        ]
        suite = [("testball", "scenes/testball", (32, 18), smoke_methods)]
    elif args.suite == "ablation":
        suite = [("testball", "scenes/testball", (512, 288), ablation_methods()),
                 ("teapot", "scenes/teapot", (512, 288), ablation_methods())]
    elif args.suite == "report":
        suite = [(name, path, res, methods_for_report(resolution=res)) for name, path, res in REPORT_SCENES]
    elif args.suite == "report_fast":
        suite = [(name, path, res, methods_for_report(reference_spp=32, include_variants=False, resolution=res))
                 for name, path, res in REPORT_SCENES]
    else:
        suite = [(name, path, (1280, 720), methods_for_report(reference_spp=32, include_variants=False, resolution=(1280, 720)))
                 for name, path, _ in REPORT_SCENES]

    if args.limit_scenes > 0:
        suite = suite[:args.limit_scenes]
    if args.scene_names:
        names = set(args.scene_names)
        suite = [item for item in suite if item[0] in names]
        if not suite:
            raise ValueError(f"no scenes selected by --scene-names {args.scene_names}")
    write_json(run_root / "experiment_plan.json", {
        "suite": args.suite,
        "commit": git_commit(),
        "scene_display_exposure": {
            name: report_exposure(name) for name, _, _, _ in suite
        },
        "scene_adjustments": {
            "green_bathroom": "Use a pinhole camera for report renders to remove the original thin-lens defocus blur."
        },
        "scenes": [
            {"name": name, "path": path, "resolution": res, "methods": [m[0] for m in methods]}
            for name, path, res, methods in suite
        ],
    })
    rows = render_suite(suite, run_root, args.timeout)
    summarize(rows, run_root)
    print(f"[done] {run_root}")


if __name__ == "__main__":
    main()
