from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROVENANCE = ROOT / "provenance"
BASE_NAME = "fig4_graph_prediction_frontier"

BLUE = "#4C78A8"
RUST = "#B45F4A"
TEAL = "#4F8C82"
CHARCOAL = "#3F454B"
MID_GREY = "#747B82"
LIGHT_GREY = "#D9DEE3"
WHITE = "#FFFFFF"

REPAIR_ORDER = [
    r"$\lambda=3\times10^{-4}$",
    r"$\lambda=10^{-3}$",
    r"$\lambda=3\times10^{-3}$",
    r"$\lambda=10^{-2}$",
]
METHOD_ORDER = [
    "Concat x-only",
    "Baseline JRNGC",
    "Full auxiliary lc10",
    *REPAIR_ORDER,
]
REPAIR_MARKERS = ["o", "s", "D", "^"]
METHOD_META = {
    "Concat x-only": {"label": "Concat", "color": MID_GREY, "marker": "o"},
    "Baseline JRNGC": {"label": "Baseline", "color": BLUE, "marker": "s"},
    "Full auxiliary lc10": {"label": "Full aux 10x", "color": TEAL, "marker": "D"},
    **{
        method: {"label": "Repair", "color": RUST, "marker": marker}
        for method, marker in zip(REPAIR_ORDER, REPAIR_MARKERS, strict=True)
    },
}
LAMBDA_LABELS = ["3e-4", "1e-3", "3e-3", "1e-2"]


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
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


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.tick_params(colors=CHARCOAL)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.4,
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
            "figure": "Fig. 5",
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
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in [*tracked, source_manifest]
        if path.is_file()
    ]
    write_text_lf(
        PROVENANCE / "CHECKSUMS.txt", "\n".join(sorted(lines))
    )


def scatter_panel(
    ax: plt.Axes,
    tradeoff: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    title: str,
    *,
    annotate: bool = False,
) -> None:
    repair_means: list[tuple[float, float]] = []
    for method in METHOD_ORDER:
        rows = tradeoff.loc[tradeoff["method"].eq(method)].sort_values("data_seed")
        meta = METHOD_META[method]
        x_values = rows["relative_mse_percent"].to_numpy(dtype=float)
        y_values = rows[metric].to_numpy(dtype=float)
        ax.scatter(
            x_values,
            y_values,
            s=12,
            marker=meta["marker"],
            facecolor=WHITE,
            edgecolor=meta["color"],
            linewidth=0.65,
            alpha=0.82,
            zorder=3,
        )
        x_mean, x_sd = mean_sd(x_values)
        y_mean, y_sd = mean_sd(y_values)
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_sd,
            yerr=y_sd,
            marker=meta["marker"],
            markersize=4.8,
            markerfacecolor=meta["color"],
            markeredgecolor=WHITE,
            markeredgewidth=0.6,
            color=meta["color"],
            linewidth=0.85,
            elinewidth=0.75,
            capsize=1.8,
            zorder=5,
        )
        if method in REPAIR_ORDER:
            repair_means.append((x_mean, y_mean))
        elif annotate:
            offsets = {
                "Concat x-only": (4, -11),
                "Baseline JRNGC": (4, 6),
                "Full auxiliary lc10": (4, -13),
            }
            ax.annotate(
                meta["label"],
                (x_mean, y_mean),
                xytext=offsets[method],
                textcoords="offset points",
                fontsize=5.7,
                color=meta["color"],
                ha="left",
                va="center",
            )
    if annotate:
        offsets = [(0, 7), (0, -11), (0, 7), (0, -11)]
        for (x_mean, y_mean), label, offset in zip(
            repair_means, LAMBDA_LABELS, offsets, strict=True
        ):
            ax.annotate(
                label,
                (x_mean, y_mean),
                xytext=offset,
                textcoords="offset points",
                fontsize=5.6,
                color=RUST,
                ha="center",
                va="bottom" if offset[1] > 0 else "top",
            )
    ax.axvline(10.0, color=MID_GREY, linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xlim(-4, 130)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel("Relative pure-MSE change (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    style_axis(ax)


def repair_summary(tradeoff: pd.DataFrame) -> pd.DataFrame:
    concat = (
        tradeoff.loc[tradeoff["method"].eq("Concat x-only")]
        .set_index("data_seed")
        .sort_index()
    )
    records: list[dict[str, float | str]] = []
    for method, label in zip(REPAIR_ORDER, LAMBDA_LABELS, strict=True):
        rows = tradeoff.loc[tradeoff["method"].eq(method)].set_index("data_seed").sort_index()
        if rows.index.tolist() != concat.index.tolist():
            raise ValueError("Repair and concat data-seed indices do not align")
        records.append(
            {
                "method": method,
                "label": label,
                "auroc_gain": float((rows["auroc"] - concat["auroc"]).mean()),
                "auroc_gain_sd": float(
                    (rows["auroc"] - concat["auroc"]).std(ddof=0)
                ),
                "auprc_gain": float((rows["auprc"] - concat["auprc"]).mean()),
                "auprc_gain_sd": float(
                    (rows["auprc"] - concat["auprc"]).std(ddof=0)
                ),
                "coefficient_gain": float(
                    (rows["coefficient_r"] - concat["coefficient_r"]).mean()
                ),
                "coefficient_gain_sd": float(
                    (rows["coefficient_r"] - concat["coefficient_r"]).std(ddof=0)
                ),
                "relative_mse_percent": float(rows["relative_mse_percent"].mean()),
                "relative_mse_percent_sd": float(
                    rows["relative_mse_percent"].std(ddof=0)
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def draw_gain_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = np.arange(len(summary), dtype=float)
    series = [
        ("auroc_gain", "auroc_gain_sd", "AUROC", BLUE, "o"),
        ("auprc_gain", "auprc_gain_sd", "AUPRC", TEAL, "s"),
        ("coefficient_gain", "coefficient_gain_sd", "Coefficient r", RUST, "D"),
    ]
    for field, spread, label, color, marker in series:
        ax.errorbar(
            x,
            summary[field],
            yerr=summary[spread],
            color=color,
            marker=marker,
            markerfacecolor=WHITE,
            markeredgecolor=color,
            markeredgewidth=0.75,
            markersize=3.8,
            linewidth=1.0,
            elinewidth=0.7,
            capsize=1.8,
            label=label,
            zorder=3,
        )
    ax.axhline(0, color=MID_GREY, linewidth=0.7)
    ax.set_xlim(-0.35, len(summary) - 0.65)
    ax.set_ylim(0, 0.34)
    ax.set_yticks([0, 0.15, 0.30])
    ax.set_xticks(x, summary["label"])
    ax.set_ylabel(r"Mean $\Delta$ versus concat")
    ax.set_title("Graph effect by strength", pad=4)
    ax.legend(loc="upper left", ncol=1, handlelength=1.6)
    style_axis(ax)


def draw_mse_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = np.arange(len(summary), dtype=float)
    ax.errorbar(
        x,
        summary["relative_mse_percent"],
        yerr=summary["relative_mse_percent_sd"],
        color=RUST,
        marker="o",
        markerfacecolor=WHITE,
        markeredgecolor=RUST,
        markeredgewidth=0.8,
        markersize=4.0,
        linewidth=1.1,
        elinewidth=0.75,
        capsize=1.8,
        zorder=3,
    )
    ax.axhline(10, color=MID_GREY, linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xlim(-0.35, len(summary) - 0.65)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 40, 80, 120])
    ax.set_xticks(x, summary["label"])
    ax.set_ylabel("Relative pure-MSE change (%)")
    ax.set_title("Prediction cost by strength", pad=4)
    style_axis(ax)


def main() -> None:
    configure_style()
    tradeoff = pd.read_csv(DATA / "repair_tradeoff.csv")
    require_columns(
        tradeoff,
        {
            "data_seed",
            "method",
            "relative_mse_percent",
            "auroc",
            "auprc",
            "coefficient_r",
            "fixed_target_prediction_mse",
        },
        "repair_tradeoff.csv",
    )
    if set(tradeoff["method"]) != set(METHOD_ORDER):
        raise ValueError("Unexpected method set")
    if not bool((tradeoff.groupby("method").size() == 3).all()):
        raise ValueError("Every method must contain three data-seed units")

    summary = repair_summary(tradeoff)
    fig = plt.figure(figsize=(7.5, 4.6))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.81,
        hspace=0.66,
    )
    upper = outer[0].subgridspec(1, 3, wspace=0.36)
    lower = outer[1].subgridspec(1, 2, wspace=0.25)
    axes = [
        fig.add_subplot(upper[0, 0]),
        fig.add_subplot(upper[0, 1]),
        fig.add_subplot(upper[0, 2]),
        fig.add_subplot(lower[0, 0]),
        fig.add_subplot(lower[0, 1]),
    ]

    scatter_panel(
        axes[0],
        tradeoff,
        "auroc",
        "Total nominal AUROC",
        (0.62, 1.01),
        [0.7, 0.85, 1.0],
        "Direct graph ranking",
    )
    scatter_panel(
        axes[1],
        tradeoff,
        "auprc",
        "Total nominal AUPRC",
        (0.50, 1.01),
        [0.5, 0.75, 1.0],
        "Edge-retrieval precision",
    )
    scatter_panel(
        axes[2],
        tradeoff,
        "coefficient_r",
        r"Lag-1 coefficient Pearson $r$",
        (0.55, 1.01),
        [0.6, 0.8, 1.0],
        "Coefficient fidelity",
    )
    draw_gain_panel(axes[3], summary)
    draw_mse_panel(axes[4], summary)

    for label, ax in zip(("a", "b", "c", "d", "e"), axes, strict=True):
        panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_META[method]["marker"],
            markerfacecolor=METHOD_META[method]["color"],
            markeredgecolor=WHITE,
            color="none",
            markersize=4.6,
            label=METHOD_META[method]["label"],
        )
        for method in ["Concat x-only", "Baseline JRNGC", "Full auxiliary lc10"]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.4,
    )
    repair_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_META[method]["marker"],
            markerfacecolor=RUST,
            markeredgecolor=WHITE,
            color="none",
            markersize=4.6,
            label=method,
        )
        for method in REPAIR_ORDER
    ]
    fig.legend(
        handles=repair_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        title="Coverage-aligned repair",
        title_fontsize=6.5,
        ncol=4,
        columnspacing=1.3,
        handletextpad=0.35,
    )

    for suffix in ("pdf", "svg"):
        output_path = ROOT / f"{BASE_NAME}.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={
                "Creator": "Python/matplotlib",
                "Title": "Figure 4 graph-prediction frontier",
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

    method_means = {
        method: {
            field: float(rows[field].mean())
            for field in (
                "relative_mse_percent",
                "auroc",
                "auprc",
                "coefficient_r",
                "fixed_target_prediction_mse",
            )
        }
        for method, rows in tradeoff.groupby("method", sort=False)
    }
    first_repair = method_means[REPAIR_ORDER[0]]
    concat = method_means["Concat x-only"]
    if not math.isclose(
        first_repair["relative_mse_percent"],
        9.77402899,
        rel_tol=0,
        abs_tol=1e-5,
    ):
        raise RuntimeError("lambda=3e-4 mean MSE change does not match the manuscript")
    if not math.isclose(
        first_repair["auroc"] - concat["auroc"],
        0.072,
        rel_tol=0,
        abs_tol=8e-4,
    ):
        raise RuntimeError("lambda=3e-4 AUROC effect does not match the manuscript")

    bounds_passed = bool(
        tradeoff["relative_mse_percent"].between(-4, 130).all()
        and tradeoff["auroc"].between(0.62, 1.01).all()
        and tradeoff["auprc"].between(0.50, 1.01).all()
        and tradeoff["coefficient_r"].between(0.55, 1.01).all()
        and summary[
            ["auroc_gain", "auprc_gain", "coefficient_gain"]
        ].to_numpy(dtype=float).min()
        >= 0
        and summary["relative_mse_percent"].between(0, 125).all()
    )
    if not bounds_passed:
        raise RuntimeError("At least one plotted value falls outside its declared axis")

    write_provenance(
        {
            "passed": True,
            "backend": "Python/matplotlib",
            "layout": "three upper panels and two wider lower panels",
            "all_source_values_finite": True,
            "all_values_within_axes": True,
            "no_observations_excluded_from_source_table": True,
            "displayed_comparators": METHOD_ORDER[:3],
            "equal_lambda_comparator_displayed": False,
            "population_sd_ddof": 0,
            "record_counts": {
                "tradeoff_rows": int(len(tradeoff)),
            },
            "method_means": method_means,
            "repair_summary": summary.to_dict(orient="records"),
            "gate_render": "not rendered; frozen outcomes are a companion LaTeX table",
            "mse_boundary_percent": 10.0,
        }
    )


if __name__ == "__main__":
    main()
