from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROVENANCE = ROOT / "provenance"
BASE_NAME = "fig2_controlled_decoupling"

BLUE = "#4C78A8"
RUST = "#B45F4A"
TEAL = "#4F8C82"
CHARCOAL = "#3F454B"
MID_GREY = "#747B82"
LIGHT_GREY = "#D9DEE3"
PALE_BLUE = "#DCE7F1"
PALE_RUST = "#EEDDD8"
WHITE = "#FFFFFF"


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or Inf")


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.tick_params(colors=CHARCOAL)
    if grid:
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.22,
        1.18,
        label,
        transform=ax.transAxes,
        fontsize=9.6,
        fontweight="bold",
        color=CHARCOAL,
        ha="left",
        va="bottom",
    )


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def write_provenance(validation: dict[str, object]) -> None:
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    validation_path = PROVENANCE / "plot_validation.json"
    write_text_lf(validation_path, json.dumps(validation, indent=2, sort_keys=True))
    tracked = [
        ROOT / "figure_contract.md",
        ROOT / "figure_spec.md",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        Path(__file__),
        *sorted(DATA.glob("*")),
        ROOT / f"{BASE_NAME}.pdf",
        ROOT / f"{BASE_NAME}.svg",
        ROOT / f"{BASE_NAME}.png",
        validation_path,
    ]
    manifest = {
        "figure": BASE_NAME,
        "backend": "Python/matplotlib",
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "style_reference": {
            "article": (
                "Tao et al., Active differentiable structure learning for "
                "clinical causal discovery, Knowledge-Based Systems 327 "
                "(2025) 114145"
            ),
            "figure": "Fig. 2",
            "reuse_level": "style-only inheritance",
        },
        "files": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in tracked
            if path.is_file()
        },
    }
    source_manifest = PROVENANCE / "SOURCE_MANIFEST.json"
    write_text_lf(source_manifest, json.dumps(manifest, indent=2, sort_keys=True))
    hash_targets = [*tracked, source_manifest]
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in hash_targets
        if path.is_file()
    ]
    write_text_lf(
        PROVENANCE / "CHECKSUMS.txt", "\n".join(sorted(lines))
    )


def draw_capacity(
    ax_mse: plt.Axes, ax_auroc: plt.Axes, capacity: pd.DataFrame
) -> None:
    d_order = [0, 1, 2, 4, 8, 16]
    for _, rows in capacity.groupby("paired_seed_index", sort=True):
        rows = rows.set_index("d_cond").loc[d_order].reset_index()
        ax_mse.plot(
            d_order,
            rows["fixed_target_prediction_mse"],
            color=PALE_BLUE,
            linewidth=0.75,
            marker="o",
            markersize=2.4,
            markerfacecolor=WHITE,
            markeredgecolor=BLUE,
            markeredgewidth=0.45,
            zorder=2,
        )
        ax_auroc.plot(
            d_order,
            rows["partial_nominal_auroc"],
            color=PALE_RUST,
            linewidth=0.75,
            marker="D",
            markersize=2.3,
            markerfacecolor=WHITE,
            markeredgecolor=RUST,
            markeredgewidth=0.45,
            zorder=2,
        )

    summary = capacity.groupby("d_cond", sort=True).mean(numeric_only=True).loc[d_order]
    ax_mse.plot(
        d_order,
        summary["fixed_target_prediction_mse"],
        color=BLUE,
        linewidth=1.45,
        marker="o",
        markersize=4.0,
        markerfacecolor=BLUE,
        markeredgecolor=WHITE,
        markeredgewidth=0.6,
        zorder=4,
    )
    ax_auroc.plot(
        d_order,
        summary["partial_nominal_auroc"],
        color=RUST,
        linewidth=1.45,
        marker="D",
        markersize=3.9,
        markerfacecolor=RUST,
        markeredgecolor=WHITE,
        markeredgewidth=0.6,
        zorder=4,
    )

    ax_mse.set_ylim(0.0010, 0.0076)
    ax_mse.set_yticks([0.002, 0.004, 0.006])
    ax_mse.set_ylabel("Fixed-target pure MSE")
    ax_mse.set_title("Prediction fit", pad=4)
    ax_mse.tick_params(axis="x", labelbottom=False)
    style_axis(ax_mse)

    ax_auroc.set_ylim(0.25, 1.03)
    ax_auroc.set_yticks([0.3, 0.6, 0.9])
    ax_auroc.set_ylabel("Partial nominal AUROC")
    ax_auroc.set_xlabel(r"Auxiliary dimension $d_{\mathrm{cond}}$")
    ax_auroc.set_xticks(d_order)
    ax_auroc.set_xticklabels([str(value) for value in d_order])
    ax_auroc.set_title("Graph ranking", pad=4)
    style_axis(ax_auroc)

    for ax in (ax_mse, ax_auroc):
        ax.set_xlim(-0.8, 16.8)


def draw_fidelity(
    ax_auroc: plt.Axes, ax_coef: plt.Axes, coefficient: pd.DataFrame
) -> None:
    categories = ["Baseline", "Concat\npartial", "Concat\ntotal"]
    colors = [BLUE, RUST, TEAL]
    markers = ["s", "o", "D"]
    metric_rows = [
        (
            ax_auroc,
            ["baseline_auroc", "concat_partial_auroc", "concat_total_auroc"],
            "Direct graph AUROC",
            (0.40, 1.02),
            [0.5, 0.75, 1.0],
        ),
        (
            ax_coef,
            [
                "baseline_coefficient_r",
                "concat_partial_coefficient_r",
                "concat_total_coefficient_r",
            ],
            r"Lag-1 coefficient Pearson $r$",
            (0.0, 1.02),
            [0.0, 0.5, 1.0],
        ),
    ]

    for ax, fields, ylabel, ylim, yticks in metric_rows:
        for _, row in coefficient.iterrows():
            values = row[fields].to_numpy(dtype=float)
            ax.plot(
                np.arange(3),
                values,
                color=LIGHT_GREY,
                linewidth=0.6,
                zorder=1,
            )
        for index, (field, color, marker) in enumerate(
            zip(fields, colors, markers, strict=True)
        ):
            values = coefficient[field].to_numpy(dtype=float)
            mean, sd = mean_sd(values)
            ax.scatter(
                np.full(len(values), index),
                values,
                s=13,
                marker=marker,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.7,
                alpha=0.85,
                zorder=3,
            )
            ax.errorbar(
                index,
                mean,
                yerr=sd,
                marker=marker,
                markersize=5.1,
                markerfacecolor=color,
                markeredgecolor=WHITE,
                markeredgewidth=0.6,
                color=color,
                linewidth=0.9,
                elinewidth=0.8,
                capsize=2.0,
                zorder=5,
            )
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(3), categories)
        style_axis(ax)

    ax_auroc.set_title("Edge ranking", pad=4)
    ax_auroc.tick_params(axis="x", labelbottom=False)
    ax_coef.set_title("Coefficient fidelity", pad=4)


def draw_interventions(
    ax_mask: plt.Axes, ax_shuffle: plt.Axes, interventions: pd.DataFrame
) -> None:
    for ax, condition, title in (
        (ax_mask, "mask", "Mask"),
        (ax_shuffle, "shuffle", "Shuffle"),
    ):
        rows = interventions.loc[interventions["condition"] == condition].sort_values(
            "data_seed"
        )
        raw = rows["raw_history_delta"].to_numpy(dtype=float)
        auxiliary = rows["auxiliary_delta"].to_numpy(dtype=float)
        for raw_value, aux_value in zip(raw, auxiliary, strict=True):
            ax.plot(
                [0, 1],
                [raw_value, aux_value],
                color=LIGHT_GREY,
                linewidth=0.65,
                zorder=1,
            )
        for index, values, color, marker in (
            (0, raw, BLUE, "o"),
            (1, auxiliary, RUST, "D"),
        ):
            mean, sd = mean_sd(values)
            ax.scatter(
                np.full(len(values), index),
                values,
                s=13,
                marker=marker,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.7,
                zorder=3,
            )
            ax.errorbar(
                index,
                mean,
                yerr=sd,
                marker=marker,
                markersize=5.1,
                markerfacecolor=color,
                markeredgecolor=WHITE,
                markeredgewidth=0.6,
                color=color,
                linewidth=0.9,
                elinewidth=0.8,
                capsize=2.0,
                zorder=5,
            )
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], ["Raw", "Auxiliary"])
        ax.set_ylim(0, 3.35)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel(r"Fixed-target $\Delta$MSE")
        ax.set_title(title, pad=4)
        style_axis(ax)
    ax_mask.tick_params(axis="x", labelbottom=False)
    ax_shuffle.set_xlabel("Perturbed route")


def main() -> None:
    configure_style()
    capacity = pd.read_csv(DATA / "capacity_replication.csv")
    coefficient = pd.read_csv(DATA / "coefficient_replication.csv")
    interventions = pd.read_csv(DATA / "fixed_target_interventions.csv")

    require_columns(
        capacity,
        {
            "paired_seed_index",
            "d_cond",
            "fixed_target_prediction_mse",
            "partial_nominal_auroc",
        },
        "capacity_replication.csv",
    )
    require_columns(
        coefficient,
        {
            "data_seed",
            "baseline_auroc",
            "concat_partial_auroc",
            "concat_total_auroc",
            "baseline_coefficient_r",
            "concat_partial_coefficient_r",
            "concat_total_coefficient_r",
        },
        "coefficient_replication.csv",
    )
    require_columns(
        interventions,
        {"data_seed", "condition", "raw_history_delta", "auxiliary_delta"},
        "fixed_target_interventions.csv",
    )

    if len(capacity) != 30 or capacity["paired_seed_index"].nunique() != 5:
        raise ValueError("Capacity replication must contain five paired six-point runs")
    if len(coefficient) != 5:
        raise ValueError("Coefficient replication must contain five paired runs")
    if len(interventions) != 10 or set(interventions["condition"]) != {
        "mask",
        "shuffle",
    }:
        raise ValueError("Intervention table must contain five mask and five shuffle pairs")

    fig = plt.figure(figsize=(7.5, 4.45))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.88,
        wspace=0.38,
        hspace=0.28,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, col]) for col in range(3)],
            [fig.add_subplot(grid[1, col]) for col in range(3)],
        ]
    )

    draw_capacity(axes[0, 0], axes[1, 0], capacity)
    draw_fidelity(axes[0, 1], axes[1, 1], coefficient)
    draw_interventions(axes[0, 2], axes[1, 2], interventions)

    column_titles = [
        "Capacity decoupling",
        "Graph and coefficient fidelity",
        "Fixed-target route use",
    ]
    for label, title, ax in zip(("a", "b", "c"), column_titles, axes[0], strict=True):
        panel_label(ax, label)
        ax.text(
            0.5,
            1.18,
            title,
            transform=ax.transAxes,
            fontsize=8.7,
            fontweight="bold",
            color=CHARCOAL,
            ha="center",
            va="bottom",
        )

    for suffix in ("pdf", "svg"):
        output_path = ROOT / f"{BASE_NAME}.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={
                "Creator": "Python/matplotlib",
                "Title": "Figure 2 controlled prediction-knowledge decoupling",
            },
        )
        if suffix == "svg":
            write_text_lf(output_path, output_path.read_text(encoding="utf-8"))
    fig.savefig(
        ROOT / f"{BASE_NAME}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)

    capacity_summary = (
        capacity.groupby("d_cond", sort=True)
        .mean(numeric_only=True)
        .loc[[0, 1, 2, 4, 8, 16]]
    )
    baseline_mse = float(capacity_summary.loc[0, "fixed_target_prediction_mse"])
    high_mse = float(capacity_summary.loc[16, "fixed_target_prediction_mse"])
    baseline_auroc = float(capacity_summary.loc[0, "partial_nominal_auroc"])
    high_auroc = float(capacity_summary.loc[16, "partial_nominal_auroc"])
    if not math.isclose(baseline_mse, 0.00648, rel_tol=0, abs_tol=5e-6):
        raise RuntimeError("Capacity baseline MSE does not match the manuscript")
    if not math.isclose(high_mse, 0.00166, rel_tol=0, abs_tol=5e-6):
        raise RuntimeError("Capacity high-dimension MSE does not match the manuscript")
    if not math.isclose(baseline_auroc, 0.877, rel_tol=0, abs_tol=5e-4):
        raise RuntimeError("Capacity baseline AUROC does not match the manuscript")
    if not math.isclose(high_auroc, 0.460, rel_tol=0, abs_tol=5e-4):
        raise RuntimeError("Capacity high-dimension AUROC does not match the manuscript")

    bounds_passed = bool(
        capacity["fixed_target_prediction_mse"].between(0.0010, 0.0076).all()
        and capacity["partial_nominal_auroc"].between(0.25, 1.03).all()
        and coefficient[
            ["baseline_auroc", "concat_partial_auroc", "concat_total_auroc"]
        ]
        .to_numpy(dtype=float)
        .min()
        >= 0.40
        and coefficient[
            [
                "baseline_coefficient_r",
                "concat_partial_coefficient_r",
                "concat_total_coefficient_r",
            ]
        ]
        .to_numpy(dtype=float)
        .min()
        >= 0.0
        and interventions[["raw_history_delta", "auxiliary_delta"]]
        .to_numpy(dtype=float)
        .max()
        <= 3.35
    )
    if not bounds_passed:
        raise RuntimeError("At least one plotted value falls outside its declared axis")

    write_provenance(
        {
            "passed": True,
            "backend": "Python/matplotlib",
            "layout": "two rows by three columns",
            "all_source_values_finite": True,
            "all_values_within_axes": True,
            "no_observations_excluded": True,
            "population_sd_ddof": 0,
            "record_counts": {
                "capacity_rows": int(len(capacity)),
                "coefficient_rows": int(len(coefficient)),
                "intervention_rows": int(len(interventions)),
            },
            "capacity_endpoints": {
                "baseline_mse": baseline_mse,
                "d16_mse": high_mse,
                "baseline_partial_auroc": baseline_auroc,
                "d16_partial_auroc": high_auroc,
            },
        }
    )


if __name__ == "__main__":
    main()
