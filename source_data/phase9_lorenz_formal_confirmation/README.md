# Phase 9 Lorenz-96 Formal Confirmation

This directory is the self-contained manuscript source for the fresh
strong-known-graph operating-point table.

- `data_seed_level_metrics.csv`: five formal data-seed analysis units; each
  row averages two frozen training seeds.
- `formal_confirmation_summary.json`: canonical aggregate decision and release
  lock.
- `frozen_config.json`: exact dataset, model, seed, training, evaluation, and
  gate configuration used by the formal release.
- `generate_table.py`: recomputes sample-SD summaries and the English/Chinese
  LaTeX tables.
- `table_summary.csv`: compact numeric source used by the manuscript and
  master numeric audit.
- `table_lorenz_confirmation_en.tex` and
  `table_lorenz_confirmation_zh.tex`: generated table fragments.
- `SOURCE_MANIFEST.json` and `CHECKSUMS.txt`: provenance records.

The formal release commit is
`2637bb798a2bd09c4f60fe3ead32abbffec3b8ca`. The archived result bundle has
checksum
`959916f42e58c61d632d5113b977a1907461b5b89991abac51b3f9baf8a53209`.

The permitted interpretation is a score-coverage case study: baseline graph
recovery is strong, the auxiliary concat model lowers pure prediction MSE, and
the x-only partial score omits substantial total raw-chain attribution. The
table does not support Mamba/ISTF superiority or a successful repair.
