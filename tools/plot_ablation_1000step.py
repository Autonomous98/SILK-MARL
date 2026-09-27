"""Plot the three formal ablation runs from their 1000-step test logs.

The labels follow the paper's ablation definitions. Spread3 was exported only
as per-step three-seed aggregates; its shaded SD is therefore smoothed from
the stored per-step SD. Spread4 and LBF2 are aggregated from per-seed records.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("exp1_exp2_exp3", "Ours", "#d83232", "-"),
    ("exp1_exp3", "w/o Credit", "#1877b8", "--"),
    ("exp2_exp3", "w/o Semantic Actions", "#008f6c", "-."),
)
TASKS = {
    "spread3": {
        "title": "Simple Spread (3 Agents)",
        "metrics": (
            ("return", "Test Return", "Test return", None),
            ("success", "Success Rate", "Success rate", (0.0, 1.0)),
            ("distance", "Mean Landmark Distance", "Mean landmark distance", None),
        ),
    },
    "spread4": {
        "title": "Simple Spread (4 Agents)",
        "metrics": (
            ("return", "Test Return", "Test return", None),
            ("success", "Success Rate", "Success rate", (0.0, 1.0)),
            ("distance", "Mean Landmark Distance", "Mean landmark distance", None),
        ),
    },
    "lbf2": {
        "title": "Level-Based Foraging (2 Agents)",
        "metrics": (
            ("return", "Test Return", "Test return", (0.0, 1.0)),
            ("success", "Success Rate", "Success rate", (0.0, 1.0)),
            ("food", "Food Collection Rate", "Food collection rate", (0.0, 1.0)),
        ),
    },
}
SOURCE_DIRS = {
    "spread3": "selected_spread3_2m_exp3_pipeline_summary",
    "spread4": "selected_spread4_2m_exp3_pipeline_summary",
    "lbf2": "selected_lbf2_2m_exp3_pipeline_summary",
}
SOURCE_METRICS = {
    "spread3": {
        "return": "test_return_mean",
        "success": "test_success_rate_mean",
        "distance": "test_mean_landmark_dist_mean",
    },
    "spread4": {
        "return": "test_return",
        "success": "success_rate",
        "distance": "mean_landmark_dist",
    },
    "lbf2": {
        "return": "test_return",
        "success": "success_rate",
        "food": "food_collection_rate",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def smooth(values: np.ndarray, width: int = 21) -> np.ndarray:
    kernel = np.ones(width, dtype=float)
    return np.convolve(values, kernel, mode="same") / np.convolve(
        np.ones_like(values), kernel, mode="same"
    )


def load_spread3(repo: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = repo / SOURCE_DIRS["spread3"] / "spread3_dense_metrics_1000step.csv"
    rows = read_csv(path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["metric"])].append(row)

    curves = {}
    for method, *_ in METHODS:
        for metric, source_metric in SOURCE_METRICS["spread3"].items():
            series = sorted(grouped[(method, source_metric)], key=lambda row: int(row["step"]))
            steps = np.array([int(row["step"]) for row in series])
            if len(series) != 2001 or not np.all(np.diff(steps) == 1000):
                raise ValueError(f"Unexpected Spread3 cadence: {method}/{metric}")
            if any(int(row["seed_count"]) != 3 for row in series):
                raise ValueError(f"Incomplete Spread3 seed coverage: {method}/{metric}")
            mean = np.array([float(row["mean"]) for row in series])
            sd = np.array([float(row["std"]) for row in series])
            curves[(method, metric)] = ((steps - steps[0]) / 1e6, mean, sd)
    return curves


def load_per_seed(
    repo: Path, task: str
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = repo / SOURCE_DIRS[task] / "formal_metrics_long.csv"
    grouped: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["tag"], row["metric"], int(row["seed"]))].append(
            (int(row["step"]), float(row["value"]))
        )

    grid = np.arange(0, 2_000_001, 1000, dtype=float)
    curves = {}
    for method, *_ in METHODS:
        for metric, source_metric in SOURCE_METRICS[task].items():
            seed_values = []
            for seed in range(3):
                samples = sorted(grouped[(method, source_metric, seed)])
                if len(samples) < 1800:
                    raise ValueError(f"Missing {task}/{method}/{metric}/seed{seed} records")
                steps = np.array([sample[0] for sample in samples], dtype=float)
                values = np.array([sample[1] for sample in samples], dtype=float)
                if not np.all(np.diff(steps) > 0):
                    raise ValueError(f"Non-monotonic {task}/{method}/{metric}/seed{seed} steps")
                nominal_steps = steps - steps[0]
                seed_values.append(np.interp(grid, nominal_steps, values, right=np.nan))
            aligned = np.stack(seed_values)
            complete = np.isfinite(aligned).all(axis=0)
            if complete.sum() < 1800:
                raise ValueError(f"Too few common {task}/{method}/{metric} points")
            common = aligned[:, complete]
            curves[(method, metric)] = (
                grid[complete] / 1e6,
                common.mean(axis=0),
                common.std(axis=0, ddof=0),
            )
    return curves


def draw_metric(ax, curves, task: str, metric: str, title: str, ylabel: str, bounds):
    all_upper = []
    all_lower = []
    lines = []
    for method, label, color, style in METHODS:
        x, raw_mean, raw_sd = curves[(method, metric)]
        mean = smooth(raw_mean)
        sd = smooth(raw_sd)
        lower, upper = mean - sd, mean + sd
        if bounds is not None:
            lower = np.clip(lower, *bounds)
            upper = np.clip(upper, *bounds)
        line, = ax.plot(x, mean, color=color, linestyle=style, linewidth=2.15, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
        all_lower.append(lower.min())
        all_upper.append(upper.max())
        lines.append(line)

    ax.set_title(title, fontweight="semibold", pad=11)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 2.0)
    ax.set_xticks(np.arange(0, 2.01, 0.5))
    if bounds is not None:
        high = min(bounds[1], max(0.08, max(all_upper) * 1.08))
        ax.set_ylim(bounds[0] - 0.02, high + 0.02)
    else:
        span = max(all_upper) - min(all_lower)
        pad = max(span * 0.06, 0.01)
        ax.set_ylim(min(all_lower) - pad, max(all_upper) + pad)
    ax.grid(True, color="#D8E0E3", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#8B989E")
    return lines


def plot_task(repo: Path, output: Path, task: str) -> None:
    curves = load_spread3(repo) if task == "spread3" else load_per_seed(repo, task)
    title = TASKS[task]["title"]
    metrics = TASKS[task]["metrics"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    for ax, (metric, heading, ylabel, bounds) in zip(axes, metrics):
        lines = draw_metric(ax, curves, task, metric, heading, ylabel, bounds)
    fig.suptitle(f"{title} - Ablation Study", fontsize=16, fontweight="semibold", y=0.98)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.24, wspace=0.29)
    fig.legend(lines, [item[1] for item in METHODS], loc="lower center",
               bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False)
    fig.text(0.5, 0.035, "3-seed mean, 21-point moving average; shading: +/- 1 SD",
             ha="center", color="#617078", fontsize=9)
    stem = output / f"{task}_ablation_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    for metric, heading, ylabel, bounds in metrics:
        fig, ax = plt.subplots(figsize=(7.3, 5.15))
        draw_metric(ax, curves, task, metric, heading, ylabel, bounds)
        fig.suptitle(f"{title} - Ablation Study", fontsize=15, fontweight="semibold", y=0.98)
        ax.legend(loc="best", frameon=False)
        fig.subplots_adjust(left=0.125, right=0.975, top=0.86, bottom=0.145)
        fig.text(0.5, 0.045, "3-seed mean, 21-point moving average; shading: +/- 1 SD",
                 ha="center", color="#617078", fontsize=8.5)
        fig.savefig(output / f"{task}_{metric}_ablation.png", dpi=300)
        plt.close(fig)

    for method, label, *_ in METHODS:
        x, mean, _ = curves[(method, "return")]
        print(f"{task:7} {label:23} {len(x):4d} points; last raw mean={mean[-1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        plot_task(args.repo, args.output, task)


if __name__ == "__main__":
    main()
