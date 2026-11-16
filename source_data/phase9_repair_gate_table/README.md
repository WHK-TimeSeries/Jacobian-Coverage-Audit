# Phase 9 repair-gate table

This directory stores the frozen gate matrix and the English/Chinese LaTeX
renderings used for the main-text repair-gate table.

- `gate_matrix.csv` is byte-identical in content to
  `source_data/figure4_gate_matrix.csv` in the frozen Phase 8 review package.
- `table_repair_gates_en.tex` and `table_repair_gates_zh.tex` render the same
  four rows without color-dependent encoding.
- The table is categorical evidence. Fig. 5 contains only continuous numeric
  panels and reads `repair_tradeoff.csv` from its independent figure directory.

No gate value or scientific conclusion was changed during the format
separation.
