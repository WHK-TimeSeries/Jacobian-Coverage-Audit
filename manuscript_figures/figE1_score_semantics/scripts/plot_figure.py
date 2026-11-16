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
BASE_NAME = "figE1_score_semantics"

RUST = "#B45F4A"
TEAL = "#4F8C82"
CHARCOAL = "#40464D"
MID_GREY = "#737A82"
LIGHT_GREY = "#D9DEE3"
WHITE = "#FFFFFF"


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.tick_params(colors=CHARCOAL)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=10,
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


def distribution_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    key: str,
    title: str,
    ylabel: str,
    color: str,
    marker: str,
    panel: str,
    *,
    legacy: bool,
) -> dict[str, float]:
    values = rows[key].to_numpy(float)
    seeds = rows["seed"].to_numpy(int)
    center = float(np.mean(values))
    sd = float(np.std(values, ddof=0))
    x_seed = np.arange(len(values), dtype=float)
    mean_x = 6.0
    ax.scatter(
        x_seed,
        values,
        s=22,
        marker=marker,
        facecolor=WHITE,
        edgecolor=color,
        linewidth=1.0,
        zorder=3,
    )
    ax.errorbar(
        mean_x,
        center,
        yerr=sd,
        marker=marker,
        markersize=6.0,
        markerfacecolor=color,
        markeredgecolor=WHITE,
        markeredgewidth=0.7,
        color=color,
        capsize=3.0,
        linewidth=1.0,
        zorder=5,
    )
    ax.set_xlim(-0.55, 6.55)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([*x_seed, mean_x], [*map(str, seeds), "mean"])
    ax.set_xlabel("Saved seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    style_axis(ax)
    panel_label(ax, panel)
    ax.text(
        0.98,
        0.05,
        rf"${center:.3f}\,\pm\,{sd:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        fontweight="semibold",
        color=color,
    )
    return {"mean": center, "population_sd": sd}


def main() -> None:
    configure_style()
    rows = pd.read_csv(DATA / "semantic_audit.csv")
    required = {
        "seed",
        "concat_partial_total_pearson",
        "legacy_filtered_raw_pearson",
        "legacy_filtered_raw_topk_jaccard",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"semantic_audit.csv is missing columns: {missing}")
    if rows["seed"].tolist() != [0, 1, 2, 3, 4]:
        raise ValueError("Expected saved seeds 0--4")
    if not np.isfinite(rows[list(required)].to_numpy(float)).all():
        raise ValueError("semantic_audit.csv contains NaN or Inf")
    metric_columns = sorted(required.difference({"seed"}))
    if not rows[metric_columns].apply(lambda column: column.between(0, 1.03).all()).all():
        raise ValueError("A semantic diagnostic lies outside the declared 0--1.03 axis")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), layout="constrained")
    summaries = {
        "concat_partial_total_pearson": distribution_panel(
            axes[0],
            rows,
            "concat_partial_total_pearson",
            "Concat route omission",
            "Partial vs total nominal Pearson",
            RUST,
            "o",
            "a",
            legacy=False,
        ),
        "legacy_filtered_raw_pearson": distribution_panel(
            axes[1],
            rows,
            "legacy_filtered_raw_pearson",
            "Legacy coordinate drift",
            "Filtered-coordinate vs raw-chain Pearson",
            TEAL,
            "D",
            "b",
            legacy=True,
        ),
        "legacy_filtered_raw_topk_jaccard": distribution_panel(
            axes[2],
            rows,
            "legacy_filtered_raw_topk_jaccard",
            "Legacy top-k stability",
            "Filtered-coordinate vs raw-chain Jaccard",
            TEAL,
            "D",
            "c",
            legacy=True,
        ),
    }

    for suffix in ("pdf", "svg"):
        output_path = ROOT / f"{BASE_NAME}.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={"Creator": "Python/matplotlib", "Title": "Figure E.1 score semantics"},
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

    if not math.isclose(summaries["concat_partial_total_pearson"]["mean"], 0.506, abs_tol=5e-4):
        raise RuntimeError("Concat partial-total mean does not match the manuscript")
    if not math.isclose(summaries["legacy_filtered_raw_pearson"]["mean"], 0.832, abs_tol=5e-4):
        raise RuntimeError("Legacy Pearson mean does not match the manuscript")
    if not math.isclose(
        summaries["legacy_filtered_raw_topk_jaccard"]["mean"], 0.784, abs_tol=5e-4
    ):
        raise RuntimeError("Legacy top-k Jaccard mean does not match the manuscript")

    write_provenance(
        {
            "passed": True,
            "backend": "Python/matplotlib",
            "all_source_values_finite": True,
            "all_values_within_axes": True,
            "no_observations_excluded": True,
            "population_sd_ddof": 0,
            "record_count": int(len(rows)),
            "summaries": summaries,
            "legacy_claim_boundary": (
                "Semantic diagnostic only; no graph-recovery, ISTF, or Mamba "
                "effectiveness claim."
            ),
        }
    )


if __name__ == "__main__":
    main()
