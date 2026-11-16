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
BASE_NAME = "fig3_heldout_audit_generality"

BLUE = "#4C78A8"
RUST = "#B45F4A"
TEAL = "#4F8C82"
CHARCOAL = "#3F454B"
MID_GREY = "#747B82"
LIGHT_GREY = "#D9DEE3"
WHITE = "#FFFFFF"

METHODS = ["mamba_concat", "tcn_concat"]
METHOD_META = {
    "mamba_concat": {"label": "Mamba concat", "color": RUST, "marker": "o"},
    "tcn_concat": {"label": "Causal TCN concat", "color": TEAL, "marker": "D"},
}
UNIT_ORDER = [
    "netsim00",
    "netsim10",
    "netsim16",
    "netsim30",
    "mocap_run_holdout",
    "mocap_salsa_holdout",
]
UNIT_LABELS = ["N0", "N10", "N16", "N30", "Run", "Salsa"]
NETSIM_UNITS = UNIT_ORDER[:4]


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.titlesize": 7.7,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.5,
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
        -0.20,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=CHARCOAL,
        ha="left",
        va="bottom",
    )


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


def plot_paired_units(
    ax: plt.Axes,
    units: pd.DataFrame,
    field: str,
    unit_names: list[str],
    *,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
    threshold: float | None = None,
    domain_labels: bool = False,
) -> None:
    positions = np.arange(len(unit_names), dtype=float)
    offset = 0.13
    for position, unit_name in zip(positions, unit_names, strict=True):
        values = []
        for method in METHODS:
            value = float(
                units.loc[
                    units["method"].eq(method)
                    & units["data_unit"].eq(unit_name),
                    field,
                ].iloc[0]
            )
            values.append(value)
        ax.plot(
            [position - offset, position + offset],
            values,
            color=LIGHT_GREY,
            linewidth=0.65,
            zorder=1,
        )
        for method, x_value, y_value in zip(
            METHODS,
            [position - offset, position + offset],
            values,
            strict=True,
        ):
            meta = METHOD_META[method]
            ax.scatter(
                x_value,
                y_value,
                s=22,
                marker=meta["marker"],
                facecolor=WHITE,
                edgecolor=meta["color"],
                linewidth=1.0,
                zorder=3,
            )
    if threshold is not None:
        ax.axhline(
            threshold,
            color=MID_GREY,
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=0,
        )
    if len(unit_names) == 6:
        ax.axvline(3.5, color=LIGHT_GREY, linewidth=0.7, zorder=0)
    ax.set_xlim(-0.55, len(unit_names) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(
        positions,
        [UNIT_LABELS[UNIT_ORDER.index(unit)] for unit in unit_names],
    )
    ax.set_title(title, pad=4)
    style_axis(ax)
    if domain_labels and len(unit_names) == 6:
        ax.text(
            1.5,
            -0.25,
            "NetSim subject",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.1,
            color=MID_GREY,
        )
        ax.text(
            4.5,
            -0.25,
            "MoCap segment",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.1,
            color=MID_GREY,
        )


def plot_support_summary(ax: plt.Axes, units: pd.DataFrame) -> None:
    fields = ["temporal_tail_median_mean", "h64_h128_mass_ratio_mean"]
    labels = ["Tail\nmass", "H64/H128\nmass ratio"]
    base_positions = np.arange(2, dtype=float)
    method_offsets = {"mamba_concat": -0.16, "tcn_concat": 0.16}
    jitter = np.linspace(-0.045, 0.045, len(UNIT_ORDER))
    for method in METHODS:
        meta = METHOD_META[method]
        rows = units.loc[units["method"].eq(method)].set_index("data_unit").loc[UNIT_ORDER]
        for field_index, field in enumerate(fields):
            values = rows[field].to_numpy(dtype=float)
            x_values = (
                base_positions[field_index]
                + method_offsets[method]
                + jitter
            )
            ax.scatter(
                x_values,
                values,
                s=13,
                marker=meta["marker"],
                facecolor=WHITE,
                edgecolor=meta["color"],
                linewidth=0.65,
                alpha=0.85,
                zorder=3,
            )
            median = float(np.median(values))
            ax.scatter(
                base_positions[field_index] + method_offsets[method],
                median,
                s=33,
                marker=meta["marker"],
                facecolor=meta["color"],
                edgecolor=WHITE,
                linewidth=0.65,
                zorder=5,
            )
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(base_positions, labels)
    ax.set_title("Measured temporal support", pad=4)
    style_axis(ax)


def main() -> None:
    configure_style()
    architecture = pd.read_csv(DATA / "stageb_architecture_summary.csv")
    units = pd.read_csv(DATA / "stageb_unit_records.csv")
    context = pd.read_csv(DATA / "stageb_graph_context.csv").iloc[0]
    decision = json.loads(
        (DATA / "stageb_confirmation_decision.json").read_text(encoding="utf-8")
    )

    require_columns(
        architecture,
        {
            "method",
            "passed_checks",
            "total_checks",
            "missing_route_qualifying_units",
            "mask_c_qualifying_units",
            "netsim_discrepancy_units",
            "h64_h128_unit_minimum",
            "nominal_horizon_max_abs",
        },
        "stageb_architecture_summary.csv",
    )
    require_columns(
        units,
        {
            "method",
            "data_unit",
            "dataset_kind",
            "missing_route_mean",
            "mask_c_delta_mean",
            "partial_total_pearson_mean",
            "partial_total_jaccard_mean",
            "coordinate_entropy_median_mean",
            "temporal_tail_median_mean",
            "h64_h128_mass_ratio_mean",
            "horizon_nominal_max_abs_mean",
        },
        "stageb_unit_records.csv",
    )
    if architecture["method"].tolist() != METHODS:
        raise ValueError("Unexpected architecture order")
    expected_pairs = {(method, unit) for method in METHODS for unit in UNIT_ORDER}
    observed_pairs = set(zip(units["method"], units["data_unit"], strict=True))
    if observed_pairs != expected_pairs:
        raise ValueError("Stage B unit grid is incomplete or contains unexpected records")
    if not units.loc[
        units["dataset_kind"].eq("mocap"), "partial_total_jaccard_mean"
    ].isna().all():
        raise ValueError("MoCap exact-top-k metric must remain unavailable")
    finite_fields = [
        "missing_route_mean",
        "mask_c_delta_mean",
        "partial_total_pearson_mean",
        "coordinate_entropy_median_mean",
        "temporal_tail_median_mean",
        "h64_h128_mass_ratio_mean",
        "horizon_nominal_max_abs_mean",
    ]
    if not np.isfinite(units[finite_fields].to_numpy(dtype=float)).all():
        raise ValueError("Stage B unit records contain NaN or Inf in plotted fields")

    fig = plt.figure(figsize=(7.5, 4.6))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.88,
        wspace=0.34,
        hspace=0.34,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    plot_paired_units(
        axes[0],
        units,
        "missing_route_mean",
        UNIT_ORDER,
        title="Missing route (6/6 each)",
        ylim=(0, 1.02),
        yticks=[0, 0.5, 1.0],
        threshold=0.20,
    )
    plot_paired_units(
        axes[1],
        units,
        "mask_c_delta_mean",
        UNIT_ORDER,
        title=r"Auxiliary use (6/6 each)",
        ylim=(0, 1.15),
        yticks=[0, 0.5, 1.0],
        threshold=0.05,
    )
    plot_paired_units(
        axes[2],
        units,
        "partial_total_pearson_mean",
        NETSIM_UNITS,
        title="Partial-total Pearson (4/4 each)",
        ylim=(0, 1.02),
        yticks=[0, 0.5, 1.0],
    )
    plot_paired_units(
        axes[3],
        units,
        "partial_total_jaccard_mean",
        NETSIM_UNITS,
        title=r"Exact-top-$k$ Jaccard",
        ylim=(0, 0.52),
        yticks=[0, 0.25, 0.5],
    )
    plot_paired_units(
        axes[4],
        units,
        "coordinate_entropy_median_mean",
        UNIT_ORDER,
        title="Source-coordinate entropy",
        ylim=(0, 1.02),
        yticks=[0, 0.5, 0.75, 1.0],
        threshold=0.75,
        domain_labels=False,
    )
    plot_support_summary(axes[5], units)

    axes[0].set_ylabel("Relative magnitude")
    axes[1].set_ylabel(r"Fixed-target $\Delta$MSE")
    axes[2].set_ylabel("Pearson correlation")
    axes[3].set_ylabel("Jaccard")
    axes[4].set_ylabel("Normalized entropy")
    axes[5].set_ylabel("Attribution mass")
    axes[3].set_xlabel("NetSim subject")

    for label, ax in zip(("a", "b", "c", "d", "e", "f"), axes, strict=True):
        panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_META[method]["marker"],
            markerfacecolor=WHITE,
            markeredgecolor=METHOD_META[method]["color"],
            markeredgewidth=0.9,
            color="none",
            markersize=4.6,
            label=METHOD_META[method]["label"],
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.45,
    )

    for suffix in ("pdf", "svg"):
        output_path = ROOT / f"{BASE_NAME}.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={
                "Creator": "Python/matplotlib",
                "Title": "Figure 3 confirmatory audit generality",
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

    for _, row in architecture.iterrows():
        if int(row["passed_checks"]) != int(row["total_checks"]):
            raise RuntimeError("Architecture confirmation checks are not all passed")
        if int(row["missing_route_qualifying_units"]) != 6:
            raise RuntimeError("Missing-route qualifying-unit count changed")
        if int(row["mask_c_qualifying_units"]) != 6:
            raise RuntimeError("Mask-c qualifying-unit count changed")
        if int(row["netsim_discrepancy_units"]) != 4:
            raise RuntimeError("NetSim discrepancy count changed")

    h64_min = float(architecture["h64_h128_unit_minimum"].min())
    nominal_max = float(architecture["nominal_horizon_max_abs"].max())
    baseline_auroc = float(context["baseline_netsim_auroc_median"])
    if h64_min < 0.9998:
        raise RuntimeError("H64/H128 stability no longer matches the frozen result")
    if nominal_max != 0.0:
        raise RuntimeError("Nominal H64/H128 score difference must remain zero")
    if not math.isclose(baseline_auroc, 0.49537037037037024, abs_tol=1e-12):
        raise RuntimeError("Baseline NetSim AUROC context changed")

    bounds_passed = bool(
        units["missing_route_mean"].between(0, 1.02).all()
        and units["mask_c_delta_mean"].between(0, 1.15).all()
        and units.loc[
            units["dataset_kind"].eq("netsim"), "partial_total_pearson_mean"
        ]
        .between(0, 1.02)
        .all()
        and units.loc[
            units["dataset_kind"].eq("netsim"), "partial_total_jaccard_mean"
        ]
        .between(0, 0.52)
        .all()
        and units["coordinate_entropy_median_mean"].between(0.94, 0.99).all()
        and units["temporal_tail_median_mean"].between(0, 1.04).all()
        and units["h64_h128_mass_ratio_mean"].between(0, 1.04).all()
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
            "record_counts": {
                "architecture_rows": int(len(architecture)),
                "unit_rows": int(len(units)),
                "netsim_graph_rows": int(units["dataset_kind"].eq("netsim").sum()),
            },
            "frozen_checks": {
                "missing_route_threshold": 0.20,
                "mask_c_threshold": 0.05,
                "h64_h128_minimum": h64_min,
                "nominal_h64_h128_max_abs": nominal_max,
                "baseline_netsim_auroc_median": baseline_auroc,
                "mass_beyond_H128_assessed": bool(
                    context["mass_beyond_H128_assessed"]
                ),
            },
            "allowed_claim": decision["allowed_claim"],
        }
    )


if __name__ == "__main__":
    main()
