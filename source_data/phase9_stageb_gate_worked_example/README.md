# Stage B gate and worked-example table

The table places the pre-specified Stage B criteria beside their outcomes
and the separate Lorenz-96 known-graph example.

Included sources in the supplementary code archive:

- `code/configs/phase9_audit_stageb_v1.json`: criteria and frozen configuration.
- `manuscript_figures/fig3_heldout_audit_generality/data/stageb_confirmation_decision.json`:
  held-out decisions and unit-level summaries.
- `source_data/phase9_lorenz_formal_confirmation/formal_confirmation_summary.json`
  and `table_summary.csv`: Lorenz-96 aggregate and table values.

The LaTeX archive contains the table and plotting inputs; model configuration
and independent execution instructions are in the supplementary code archive.

The Stage B criterion column applies only to the two Stage B result columns.
Their Pearson medians use six units; the main-text NetSim comparison uses four
subjects (0.431/0.769). Lorenz uses its own frozen conditions: entropy >=0.75
and tail >=0.05 in at least 4/5 data seeds; H64/H128 mass ratio >=0.95 and
nominal difference <=1e-7 in all 5/5. All four Lorenz conditions passed in 5/5.
The presentation table restores these existing diagnostics without changing
any source result or decision threshold.

Stage B execution release: `0397e8af27c4f396d7713b129e0d7307da732681`.
Preceding protocol lock: `21635836759d36cd2206afce85b4c15b1836e287`.
These are source identifiers, not installation paths. The thresholds are
setting-specific and are not universal defaults.
