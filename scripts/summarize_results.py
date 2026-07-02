import argparse
import csv
from pathlib import Path


def read_metrics(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def latest_run(experiment_dir):
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()

    rows = []
    for experiment_dir in sorted(args.results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        run_dir = latest_run(experiment_dir)
        if not run_dir:
            continue
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        for row in read_metrics(metrics_path):
            row["experiment"] = experiment_dir.name
            row["run"] = run_dir.name
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "run", "method", "spp", "render_seconds", "mse", "mae", "psnr", "image"]
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.output.with_suffix(".md")
    with markdown_path.open("w") as file:
        file.write("| experiment | method | spp | seconds | mse | mae | psnr |\n")
        file.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            file.write(
                f"| {row['experiment']} | {row['method']} | {row['spp']} | "
                f"{float(row['render_seconds']):.4f} | {float(row['mse']):.6g} | "
                f"{float(row['mae']):.6g} | {row['psnr']} |\n"
            )

    print(args.output)
    print(markdown_path)


if __name__ == "__main__":
    main()
