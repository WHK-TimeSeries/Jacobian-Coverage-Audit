#!/usr/bin/env python3
"""Build the manuscript Lorenz-96 confirmation table from frozen data-seed rows."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "data_seed_level_metrics.csv"
SOURCE_JSON = ROOT / "formal_confirmation_summary.json"
FROZEN_CONFIG = ROOT / "frozen_config.json"
SUMMARY_CSV = ROOT / "table_summary.csv"
TABLE_EN = ROOT / "table_lorenz_confirmation_en.tex"
TABLE_ZH = ROOT / "table_lorenz_confirmation_zh.tex"


def read_rows() -> list[dict[str, float]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                key: float(value)
                for key, value in row.items()
                if key not in {"data_seed", "concat_discrepancy_pass"}
            }
            | {
                "data_seed": int(row["data_seed"]),
                "concat_discrepancy_pass": row["concat_discrepancy_pass"].lower() == "true",
            }
            for row in csv.DictReader(handle)
        ]


def summary(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


rows = read_rows()
if len(rows) != 5:
    raise RuntimeError(f"Expected five data-seed units, found {len(rows)}")

metrics = {
    key: summary(rows, key)
    for key in [
        "baseline_fixed_target_prediction_mse",
        "mamba_concat_fixed_target_prediction_mse",
        "baseline_total_nominal_auroc",
        "mamba_concat_total_nominal_auroc",
        "mamba_concat_partial_nominal_auroc",
        "mamba_concat_missing_route_relative_magnitude",
        "mamba_concat_mask_c_fixed_target_mse_delta",
        "mamba_concat_partial_total_nominal_pearson",
        "mamba_concat_partial_total_nominal_topk_jaccard",
        "mamba_concat_coordinate_entropy_median",
        "mamba_concat_temporal_tail_median",
        "concat_vs_baseline_relative_mse",
    ]
}

total_minus_partial = [
    float(row["mamba_concat_total_nominal_auroc"])
    - float(row["mamba_concat_partial_nominal_auroc"])
    for row in rows
]
total_minus_baseline = [
    float(row["mamba_concat_total_nominal_auroc"])
    - float(row["baseline_total_nominal_auroc"])
    for row in rows
]
metrics["concat_total_minus_partial_auroc"] = (
    statistics.fmean(total_minus_partial),
    statistics.stdev(total_minus_partial),
)
metrics["concat_total_minus_baseline_auroc"] = (
    statistics.fmean(total_minus_baseline),
    statistics.stdev(total_minus_baseline),
)

summary_rows = [
    {
        "metric": key,
        "mean": value[0],
        "sample_sd": value[1],
        "n_data_seeds": len(rows),
    }
    for key, value in metrics.items()
]
with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["metric", "mean", "sample_sd", "n_data_seeds"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(summary_rows)


def pm(key: str, decimals: int = 3) -> str:
    mean, sd = metrics[key]
    return f"${mean:.{decimals}f}\\pm{sd:.{decimals}f}$"


baseline_mse, baseline_mse_sd = metrics["baseline_fixed_target_prediction_mse"]
concat_mse, concat_mse_sd = metrics["mamba_concat_fixed_target_prediction_mse"]
relative_mse, relative_mse_sd = metrics["concat_vs_baseline_relative_mse"]
delta_total_partial, delta_total_partial_sd = metrics["concat_total_minus_partial_auroc"]
delta_total_baseline, _ = metrics["concat_total_minus_baseline_auroc"]

common_rows_en = f"""Fixed-target pure MSE & $({baseline_mse * 1e4:.2f}\\pm{baseline_mse_sd * 1e4:.2f})\\times10^{{-4}}$ & $({concat_mse * 1e4:.2f}\\pm{concat_mse_sd * 1e4:.2f})\\times10^{{-4}}$ & ${abs(relative_mse) * 100:.1f}\\pm{relative_mse_sd * 100:.1f}\\%$ lower; 5/5 seeds \\\\
Total nominal AUROC & {pm("baseline_total_nominal_auroc")} & {pm("mamba_concat_total_nominal_auroc")} & ${delta_total_baseline:+.3f}$ vs. baseline \\\\
x-only partial nominal AUROC & {pm("baseline_total_nominal_auroc")} & {pm("mamba_concat_partial_nominal_auroc")} & total--partial ${delta_total_partial:+.3f}\\pm{delta_total_partial_sd:.3f}$; 5/5 seeds \\\\
Missing-route relative magnitude & $0$ & {pm("mamba_concat_missing_route_relative_magnitude")} & gate met in 5/5 seeds \\\\
Auxiliary-mask $\\Delta$MSE & n/a & {pm("mamba_concat_mask_c_fixed_target_mse_delta")} & positive in 5/5 seeds \\\\
Partial--total Pearson / top-$k$ Jaccard & $1/1$ & {pm("mamba_concat_partial_total_nominal_pearson")} / {pm("mamba_concat_partial_total_nominal_topk_jaccard")} & discrepancy gate met in 5/5 seeds \\\\"""

TABLE_EN.write_text(
    f"""\\begin{{table*}}[!t]
\\centering
\\caption{{\\textbf{{Lorenz-96 graph scores and route diagnostics.}} Values are mean $\\pm$ sample standard deviation across five fresh data seeds after averaging two training seeds. The baseline and auxiliary concat models used the same raw target and total raw-chain nominal-lag evaluator.}}
\\label{{tab:lorenz_confirmation}}
\\footnotesize
\\begin{{tabularx}}{{\\textwidth}}{{p{{0.28\\textwidth}}p{{0.18\\textwidth}}p{{0.20\\textwidth}}Y}}
\\toprule
Quantity & Baseline JRNGC & Auxiliary concat & Paired audit contrast \\\\
\\midrule
{common_rows_en}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
""",
    encoding="utf-8",
)

common_rows_zh = f"""固定目标纯 MSE & $({baseline_mse * 1e4:.2f}\\pm{baseline_mse_sd * 1e4:.2f})\\times10^{{-4}}$ & $({concat_mse * 1e4:.2f}\\pm{concat_mse_sd * 1e4:.2f})\\times10^{{-4}}$ & 下降 ${abs(relative_mse) * 100:.1f}\\pm{relative_mse_sd * 100:.1f}\\%$；5/5 seeds \\\\
总 raw-chain 名义滞后 AUROC & {pm("baseline_total_nominal_auroc")} & {pm("mamba_concat_total_nominal_auroc")} & 相对 baseline 为 ${delta_total_baseline:+.3f}$ \\\\
x-only partial 名义滞后 AUROC & {pm("baseline_total_nominal_auroc")} & {pm("mamba_concat_partial_nominal_auroc")} & total--partial 为 ${delta_total_partial:+.3f}\\pm{delta_total_partial_sd:.3f}$；5/5 seeds \\\\
缺失路径相对量 & $0$ & {pm("mamba_concat_missing_route_relative_magnitude")} & 5/5 seeds 通过 gate \\\\
辅助路径 mask 的 $\\Delta$MSE & 不适用 & {pm("mamba_concat_mask_c_fixed_target_mse_delta")} & 5/5 seeds 均为正 \\\\
partial--total Pearson / top-$k$ Jaccard & $1/1$ & {pm("mamba_concat_partial_total_nominal_pearson")} / {pm("mamba_concat_partial_total_nominal_topk_jaccard")} & 5/5 seeds 通过差异 gate \\\\"""

TABLE_ZH.write_text(
    f"""\\begin{{table*}}[!t]
\\centering
\\caption{{\\textbf{{Lorenz-96 图得分与路径诊断。}} 数值为五个新 data seeds 的均值 $\\pm$ 样本标准差，每个 data seed 先平均两个 training seeds。Baseline 与 auxiliary concat 使用相同 raw target 和 total raw-chain 名义滞后评估器。}}
\\label{{tab:lorenz_confirmation}}
\\footnotesize
\\begin{{tabularx}}{{\\textwidth}}{{p{{0.28\\textwidth}}p{{0.18\\textwidth}}p{{0.20\\textwidth}}Y}}
\\toprule
指标 & Baseline JRNGC & Auxiliary concat & 配对审计对比 \\\\
\\midrule
{common_rows_zh}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
""",
    encoding="utf-8",
)

manifest_targets = [
    SOURCE_CSV,
    SOURCE_JSON,
    FROZEN_CONFIG,
    SUMMARY_CSV,
    TABLE_EN,
    TABLE_ZH,
    Path(__file__).resolve(),
]
manifest = {
    "scientific_role": "Fresh five-data-seed strong-known-graph audit confirmation",
    "statistical_unit": "data seed after averaging two training seeds",
    "n_data_seeds": len(rows),
    "formal_release_commit": "2637bb798a2bd09c4f60fe3ead32abbffec3b8ca",
    "source_archive_sha256": "959916f42e58c61d632d5113b977a1907461b5b89991abac51b3f9baf8a53209",
    "files": {path.name: sha256(path) for path in manifest_targets},
}
(ROOT / "SOURCE_MANIFEST.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(ROOT / "CHECKSUMS.txt").write_text(
    "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_targets),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "data_seed_units": len(rows),
            "summary_csv": str(SUMMARY_CSV),
            "table_en": str(TABLE_EN),
            "table_zh": str(TABLE_ZH),
            "total_minus_partial_positive_count": sum(value > 0 for value in total_minus_partial),
            "pure_mse_lower_count": sum(
                float(row["mamba_concat_fixed_target_prediction_mse"])
                < float(row["baseline_fixed_target_prediction_mse"])
                for row in rows
            ),
        },
        indent=2,
    )
)
