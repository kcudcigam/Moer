import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

from image_metrics import metrics


def repo_root():
    return Path(__file__).resolve().parents[1]


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def copy_scene(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def deep_update(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def actual_hdr_path(scene_dir, output_file):
    path = Path(output_file)
    return Path(str(path) + ".hdr")


def run_method(root, executable, base_scene, run_dir, experiment, method):
    scene_dir = run_dir / "scene"
    copy_scene(base_scene, scene_dir)

    scene_json_path = scene_dir / "scene.json"
    scene = json.loads(scene_json_path.read_text())

    if "resolution" in experiment:
        scene["camera"]["resolution"] = experiment["resolution"]

    renderer = scene.setdefault("renderer", {})
    renderer["spp"] = method["spp"]
    renderer["output_file"] = str((run_dir / method["name"]).resolve())
    deep_update(renderer, method.get("renderer", {}))

    scene_json_path.write_text(json.dumps(scene, indent=2))

    started = time.perf_counter()
    process = subprocess.run(
        [str(executable), str(scene_dir)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started

    (run_dir / "stdout.txt").write_text(process.stdout)
    (run_dir / "stderr.txt").write_text(process.stderr)
    (run_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    if process.returncode != 0:
        raise RuntimeError(f"{method['name']} failed with {process.returncode}\n{process.stdout}\n{process.stderr}")

    image_path = actual_hdr_path(scene_dir, renderer["output_file"])
    final_image = run_dir / f"{method['name']}.hdr"
    if image_path.exists():
        if image_path.resolve() != final_image.resolve():
            for attempt in range(20):
                try:
                    shutil.copy2(image_path, final_image)
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.1)
    else:
        raise FileNotFoundError(f"missing render output: {image_path}")

    return {
        "method": method["name"],
        "spp": method["spp"],
        "render_seconds": elapsed,
        "image": final_image.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--executable", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    config = json.loads(args.config.read_text())
    executable = args.executable or (root / "target" / "bin" / "Moer_r.exe")
    output_root = args.output_root or (root / "results")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = output_root / config["name"] / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
    (experiment_dir / "git_commit.txt").write_text(git_commit(root) + "\n")

    base_scene = root / config["scene"]
    rows = []
    for method in config["methods"]:
        method_dir = experiment_dir / method["name"]
        method_dir.mkdir(parents=True, exist_ok=True)
        rows.append(run_method(root, executable, base_scene, method_dir, config, method))

    reference = next((row for row in rows if row["method"] == "path_reference"), None)
    if reference:
        reference_image = experiment_dir / reference["method"] / reference["image"]
        for row in rows:
            image = experiment_dir / row["method"] / row["image"]
            row.update(metrics(image, reference_image))

    with (experiment_dir / "metrics.csv").open("w", newline="") as file:
        fieldnames = ["method", "spp", "render_seconds", "image", "mse", "mae", "psnr"]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (experiment_dir / "timing.csv").open("w", newline="") as file:
        fieldnames = ["method", "spp", "render_seconds"]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(experiment_dir)


if __name__ == "__main__":
    main()
