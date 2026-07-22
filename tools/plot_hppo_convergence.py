import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_keys(rows, exclude=()):
    keys = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key in exclude:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                if key not in keys:
                    keys.append(key)
    return keys


def series(rows, key):
    xs, ys = [], []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        episode = row.get("episode", idx)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            xs.append(float(episode))
            ys.append(float(value))
    return xs, ys


def moving_average(values, window):
    if len(values) < 2:
        return values[:]
    window = max(1, min(window, len(values)))
    out, queue, acc = [], [], 0.0
    for value in values:
        queue.append(value)
        acc += value
        if len(queue) > window:
            acc -= queue.pop(0)
        out.append(acc / len(queue))
    return out


def safe_name(key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")


def label(key):
    return key.replace("_", " ")


def plot_one(rows, key, path, title_prefix, smooth_window=None, marker=False):
    xs, ys = series(rows, key)
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if smooth_window and len(ys) >= smooth_window:
        ax.plot(xs, ys, color="#1f77b4", alpha=0.22, linewidth=0.8, label="raw")
        ax.plot(
            xs,
            moving_average(ys, smooth_window),
            color="#d62728",
            linewidth=1.8,
            label=f"moving avg ({smooth_window})",
        )
        ax.legend(frameon=False)
    else:
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.4, marker="o" if marker else None, markersize=3)
    ax.set_title(f"{title_prefix}: {label(key)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(label(key))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_group(rows, keys, path, title, smooth_window=None):
    keys = [key for key in keys if series(rows, key)[0]]
    if not keys:
        return False
    cols = 2 if len(keys) > 1 else 1
    rows_n = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(8.4 * cols, 3.2 * rows_n), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        xs, ys = series(rows, key)
        if smooth_window and len(ys) >= smooth_window:
            ax.plot(xs, ys, alpha=0.20, linewidth=0.7)
            ax.plot(xs, moving_average(ys, smooth_window), linewidth=1.5)
        else:
            ax.plot(xs, ys, linewidth=1.2, marker="o" if len(ys) <= 30 else None, markersize=2.5)
        ax.set_title(label(key), fontsize=9)
        ax.set_xlabel("Episode")
    for ax in axes.flat[len(keys) :]:
        ax.axis("off")
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path)
    plt.close(fig)
    return True


def group_train_keys(keys):
    high_loss = [
        key
        for key in keys
        if key.startswith("high_") and any(part in key for part in ["loss", "entropy", "clip", "return"])
    ]
    low_loss = [
        key
        for key in keys
        if key.startswith("low_") and any(part in key for part in ["loss", "entropy", "clip", "return"])
    ]
    return {
        "01_core_performance": [
            "total_reward",
            "total_energy",
            "makespan",
            "response_time",
            "waiting_time",
            "avg_server_utilization",
            "deadline_success_rate",
            "avg_workflow_reliability",
            "backup_replica_count",
        ],
        "02_time_metrics": [
            key
            for key in keys
            if any(
                part in key
                for part in [
                    "makespan",
                    "span",
                    "response_time",
                    "waiting_time",
                    "task_total_time",
                    "communication_time",
                    "execution_time",
                ]
            )
        ],
        "03_deadline_reliability": [key for key in keys if "deadline" in key or "reliability" in key],
        "04_energy_backup_queue": [
            key
            for key in keys
            if any(part in key for part in ["energy", "backup", "primary", "replica", "queue", "utilization"])
        ],
        "05_high_losses": high_loss,
        "06_high_constraints": [
            key
            for key in keys
            if key.startswith("high_")
            and any(
                part in key
                for part in ["constraint", "violation", "kappa", "J_", "active", "penalty", "relu", "surrogate", "usage"]
            )
        ],
        "07_high_policy_stats": [
            key
            for key in keys
            if key.startswith("high_")
            and key not in high_loss
            and not any(
                part in key
                for part in ["constraint", "violation", "kappa", "J_", "active", "penalty", "relu", "surrogate", "usage"]
            )
        ],
        "08_low_losses": low_loss,
        "09_low_policy_stats": [key for key in keys if key.startswith("low_") and key not in low_loss],
        "10_reward_decomposition": [
            key
            for key in keys
            if "reward" in key or key in ["mean_high_reward", "mean_low_reward", "total_high_reward", "total_low_reward"]
        ],
    }


def group_eval_keys(keys):
    return {
        "01_eval_core_performance": [
            "total_reward",
            "total_energy",
            "makespan",
            "response_time",
            "waiting_time",
            "avg_server_utilization",
            "deadline_success_rate",
            "avg_workflow_reliability",
            "backup_replica_count",
        ],
        "02_eval_time_metrics": [
            key
            for key in keys
            if any(
                part in key
                for part in [
                    "makespan",
                    "span",
                    "response_time",
                    "waiting_time",
                    "task_total_time",
                    "communication_time",
                    "execution_time",
                ]
            )
        ],
        "03_eval_deadline_reliability": [key for key in keys if "deadline" in key or "reliability" in key],
        "04_eval_energy_backup": [
            key for key in keys if any(part in key for part in ["energy", "backup", "utilization"])
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    base = args.result_dir
    out = base / "convergence_plots"
    train_out = out / "train_individual_png"
    eval_out = out / "eval_individual_png"
    combined_out = out / "combined_common_png"
    for path in [out, train_out, eval_out, combined_out]:
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    train = load_json(base / "train_history.json")
    eval_rows = load_json(base / "eval_history.json")
    train_keys = numeric_keys(train, exclude={"episode", "seed"})
    eval_keys = numeric_keys(eval_rows, exclude={"episode", "eval_episodes"})
    common = [key for key in train_keys if key in eval_keys]
    generated = []

    for key in train_keys:
        path = train_out / f"{safe_name(key)}.png"
        if plot_one(train, key, path, "Train convergence", smooth_window=args.smooth_window):
            generated.append(path)
    for key in eval_keys:
        path = eval_out / f"{safe_name(key)}.png"
        if plot_one(eval_rows, key, path, "Eval convergence", marker=True):
            generated.append(path)

    for key in common:
        tx, ty = series(train, key)
        ex, ey = series(eval_rows, key)
        if not tx or not ex:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.plot(tx, ty, color="#1f77b4", alpha=0.18, linewidth=0.8, label="train raw")
        ax.plot(
            tx,
            moving_average(ty, args.smooth_window),
            color="#1f77b4",
            linewidth=1.5,
            label=f"train moving avg ({args.smooth_window})",
        )
        ax.plot(ex, ey, color="#ff7f0e", marker="o", linewidth=1.6, markersize=3.5, label="eval")
        ax.set_title(f"Train vs Eval: {label(key)}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(label(key))
        ax.legend(frameon=False)
        fig.tight_layout()
        path = combined_out / f"{safe_name(key)}.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

    for name, keys in group_train_keys(train_keys).items():
        path = out / f"{name}.png"
        if plot_group(train, [key for key in keys if key in train_keys], path, name, smooth_window=args.smooth_window):
            generated.append(path)
    for name, keys in group_eval_keys(eval_keys).items():
        path = out / f"{name}.png"
        if plot_group(eval_rows, [key for key in keys if key in eval_keys], path, name):
            generated.append(path)

    pdf_jobs = [
        ("train_all_metrics.pdf", train, train_keys, "Train convergence", args.smooth_window),
        ("eval_all_metrics.pdf", eval_rows, eval_keys, "Eval convergence", None),
        ("combined_common_metrics.pdf", None, common, "Train vs Eval", None),
    ]
    for pdf_name, rows, keys, title, smooth in pdf_jobs:
        pdf_path = out / pdf_name
        with PdfPages(pdf_path) as pdf:
            for key in keys:
                if rows is None:
                    tx, ty = series(train, key)
                    ex, ey = series(eval_rows, key)
                    if not tx or not ex:
                        continue
                    fig, ax = plt.subplots(figsize=(8.2, 4.6))
                    ax.plot(tx, ty, alpha=0.18, linewidth=0.8, label="train raw")
                    ax.plot(
                        tx,
                        moving_average(ty, args.smooth_window),
                        linewidth=1.5,
                        label=f"train moving avg ({args.smooth_window})",
                    )
                    ax.plot(ex, ey, marker="o", linewidth=1.6, markersize=3.5, label="eval")
                    ax.legend(frameon=False)
                else:
                    xs, ys = series(rows, key)
                    if not xs:
                        continue
                    fig, ax = plt.subplots(figsize=(8.2, 4.6))
                    if smooth and len(ys) >= smooth:
                        ax.plot(xs, ys, alpha=0.22, linewidth=0.8, label="raw")
                        ax.plot(xs, moving_average(ys, smooth), linewidth=1.7, label=f"moving avg ({smooth})")
                        ax.legend(frameon=False)
                    else:
                        ax.plot(xs, ys, marker="o" if len(ys) <= 30 else None, markersize=3, linewidth=1.3)
                ax.set_title(f"{title}: {label(key)}")
                ax.set_xlabel("Episode")
                ax.set_ylabel(label(key))
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        generated.append(pdf_path)

    (out / "matplotlib_generation.txt").write_text(
        "\n".join(
            [
                "Generated with conda env py310",
                "python: D:/miniconda/envs/py310/python.exe",
                f"train metrics: {len(train_keys)}",
                f"eval metrics: {len(eval_keys)}",
                f"common metrics: {len(common)}",
                f"png/pdf files generated: {len(generated)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(generated)} PNG/PDF files under {out}")
    print(f"Train metrics: {len(train_keys)}; Eval metrics: {len(eval_keys)}; Common metrics: {len(common)}")


if __name__ == "__main__":
    main()
