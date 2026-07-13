import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp", REPO / "scripts" / "run_nrc_report_experiments.py")
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


SCENES = {
    "testball": "scenes/testball",
    "teapot": "scenes/teapot",
    "disney": "scenes/disney-bsdf",
    "classroom": "scenes/classroom",
    "green_bathroom": "scenes/green-bathroom",
}


def nrc_alignment_base(resolution, quality):
    settings = EXP.scaled_nrc_for_resolution(resolution)
    if quality == "quick":
        settings.update({
            "training_spp": 8,
            "final_train_steps": 1024,
            "final_train_epochs": 8.0,
            "target_samples": 4,
            "max_training_samples": 65536,
        })
    settings.update({
        "cache_non_diffuse_surfaces": True,
        "use_batch_query": True,
        "query_bounce": 1,
    })
    return settings


def methods(resolution, nrc_spp, reference_spp, quality):
    current = nrc_alignment_base(resolution, quality)
    current["query_semantics"] = "current_vertex"
    post = dict(current)
    post["query_semantics"] = "post_scatter"
    paper = dict(post)
    paper["tcnn_encoding"] = "paper"
    return [
        ("path_low", 4, {"mode": "path_trace", "query_bounce": 1}),
        ("nrc_current_vertex", nrc_spp, current),
        ("nrc_post_scatter", nrc_spp, post),
        ("nrc_post_scatter_paper", nrc_spp, paper),
        ("path_reference", reference_spp, {"mode": "path_trace", "query_bounce": 1}),
    ]


def parse_resolution(text):
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, bbox[2] + 16, bbox[3] + 16], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)


def write_alignment_comparison(run_root, scene, method_names):
    pngs = []
    for method in method_names:
        path = run_root / "png" / scene / f"{method}.png"
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        add_label(image, method)
        pngs.append(image)
    if len(pngs) < 2:
        return

    height = min(288, min(image.height for image in pngs))
    resized = [
        image.resize((int(image.width * height / image.height), height), Image.Resampling.LANCZOS)
        for image in pngs
    ]
    out = Image.new("RGB", (sum(image.width for image in resized), height))
    x = 0
    for image in resized:
        out.paste(image, (x, 0))
        x += image.width
    out_path = run_root / "comparisons" / f"{scene}_alignment_all.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENES), default="disney")
    parser.add_argument("--resolution", default="512x288")
    parser.add_argument("--nrc-spp", type=int, default=16)
    parser.add_argument("--reference-spp", type=int, default=32)
    parser.add_argument("--quality", choices=["quick", "best"], default="quick")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--run-root", default="")
    args = parser.parse_args()

    resolution = parse_resolution(args.resolution)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.run_root) if args.run_root else EXP.RESULTS / f"alignment_{args.scene}_{args.resolution}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    scene_methods = methods(resolution, args.nrc_spp, args.reference_spp, args.quality)
    suite = [(args.scene, SCENES[args.scene], resolution, scene_methods)]
    EXP.write_json(run_root / "experiment_plan.json", {
        "suite": "alignment_ablation",
        "commit": EXP.git_commit(),
        "scene": args.scene,
        "resolution": resolution,
        "quality": args.quality,
        "methods": [name for name, _, _ in scene_methods],
    })
    rows = EXP.render_suite(suite, run_root, args.timeout)
    EXP.summarize(rows, run_root)
    write_alignment_comparison(run_root, args.scene, [name for name, _, _ in scene_methods])
    print(f"[done] {run_root}", flush=True)


if __name__ == "__main__":
    main()
