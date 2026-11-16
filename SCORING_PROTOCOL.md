# Scoring Protocol and Source Interpretation

This scientific companion explains how the submitted manuscript interprets
the included computations. Original model, gradient, and metric implementations
are unchanged. Paths below are relative to this supplementary archive.

## Graph and diagnostic objects

- Stored derivatives use `[target, source, lag]`. Absolute values are taken per
  evaluation window, then averaged over eligible windows. The primary graph
  summary takes the maximum only over the declared nominal lags.
- Raw coordinates mean observed variables, after any fixed per-variable
  standardization in that study. The scaler is held constant during autograd.
  This is not differentiation through the estimated mean or variance.
- Partial nominal attribution holds the computed auxiliary history fixed.
  Total nominal attribution includes its raw-input chain. A broader-history
  maximum is a separate predictive-attribution object, not the direct graph.
- Fixed-target MSE is in-sample one-step prediction MSE on the fitted trajectory,
  excluding every Jacobian term. Held-out subjects refer to development versus
  confirmation units, not a forecasting train/test split.
- AUPRC is trapezoidal precision-recall area in the canonical metric adapter,
  not average precision. Off-diagonal exact-top-k uses the true edge count;
  ties sort by decreasing score, target index, then source index, and returned
  edges are `(source, target)`. Coefficient Pearson compares lag-1 attribution
  with `abs(A)`, not signed coefficients.
- Missing-route magnitude uses the signed total-minus-partial derivative
  before absolute aggregation. It can exceed one through cancellation in the
  total derivative; it is not a probability. Entropy depends on coordinate scale.

## Partial-penalty normalization

For baseline and concat comparators, the partial regularizer is the mean
absolute Jacobian over sampled windows, all outputs, all raw sources, and
nominal lags. Equivalently it is the window mean of the absolute-derivative
sum divided by d_output * d_source * K. The implementation uses the first
100 eligible windows, or all windows when fewer are available. Full auxiliary
penalties weight the raw-block and auxiliary-block means separately.
These definitions agree with Equation (2) and Appendix E. The broader
full-prefix repair has its separately declared estimator and normalization;
it is not a mean over all prefix coordinates.

## Finite TCN support and archived labels

`code/src/phase9_audit_generalization.py` defines `AuxiliaryTCNPreprocessor`.
Its `TCNBlock` in `code/src/minimal_mamba.py` has a width-3, dilation-2 causal
convolution. The remaining normalization, projections, activation, and residual
operations act within time points. Thus preprocessor support is R=5 and the
K=3 predictor has raw support no greater than K+R-1=7, under the fixed scaler.

| Object | Interpretation |
| --- | --- |
| TCN total raw chain, H>=7 | Architectural temporal support is covered. This alone does not establish every audit dimension or causal recovery. |
| TCN nominal K=3 score | Older positions in the seven-lag envelope are outside this score. Partial x-only scoring still omits the auxiliary route. |
| Mamba bounded-H score | No finite support bound is established. H64/H128 agreement does not show that earlier history is zero. |
| Archived HORIZON-TRUNCATED field | The sampled evaluator assigned this generic label to all sampled-H reports; it is not an architecture-specific support conclusion. |

The generic labels originate in `SampledAuditResult.summary()` and
`build_audit_profile()`, both in `code/src/phase9_audit_generalization.py`.
They are preserved in original
reports. The separate generic framework in
`code/src/jacobian_coverage_audit.py` accepts an explicit required support
horizon. No evaluator behavior is changed by this interpretation note.
The seven-lag conclusion is a static architectural bound, not an exhaustive
numerical verification over all possible inputs.

## Study-specific provenance and aggregation

- Figure 3: `manuscript_figures/fig2_controlled_decoupling/data/` and
  `source_data/coverage_audit_submission_v4/`. Capacity, coefficient fidelity,
  and route ablations are separate settings (manuscript Table E.1). Each uses
  five paired data/model runs with population standard deviations.
- Table B.1: `source_data/supplement/tables/full_aux_penalty_all_variants.tex`
  contains the frozen five-seed full auxiliary-penalty summary. VAR(1) has
  d=8, T=500, K=1; data seed=100*s+42 for model seed s=0,...,4. Predictors use
  three width-32 residual blocks and concat has d_cond=4, d_state=4. Adam at
  0.001 runs at most 2,000 iterations, checking/restoring the best total
  objective every 50 iterations. Its graph score is partial nominal; the
  field `pred_loss` is the checked total objective, not pure MSE. Figure 5
  instead uses total nominal scoring and final checkpoints from a separate
  paired-seed study; these are not interchangeable comparator measurements.
- Table C.1: the formal Stage 1a source release is
  `65e6ae9afef552c84d8211a9d6e9aa70db48c276`, with source-manifest checksum
  `be91dc2d3ee916690ebcd519d42811f3d3698ef4eddf053d96cdf96d0f4cab3d`.
  Appendix C summarizes the study; `protocols/Supplementary_Protocol_S1.md`
  gives the D2 generator, training protocol and pre-specified decision criteria.
  Data seeds 1-3, two training seeds, and four methods/cells yield 96 formal
  runs. The statistical unit is the data seed after averaging model seeds.
  Sample standard deviations are across those three units. Historical formal
  execution outputs are not all redistributed in this curated archive.
- Figure D.1: `manuscript_figures/figE1_score_semantics/data/` contains the
  five-seed diagnostic summaries. The setting is d=6, T=100, K=3 and 120
  training iterations, with eight common window starts
  [0,13,27,41,54,68,82,96]. The legacy filter used filtered targets. The
  archived field `raw_chain_leakage_mean` (mean 0.0835) is the fraction of
  gradient mass outside the nominal time window, not cross-variable leakage.
  The derivative comparisons do not validate legacy graph performance.
- Figure 4 / Table E.2: `manuscript_figures/fig3_heldout_audit_generality/data/`.
  Two training replicates are averaged per unit. Four-NetSim median Pearson
  values are 0.431/0.769 (Mamba/TCN); six-unit medians, including the two MoCap
  segments, are 0.480/0.779. Frozen gates use their original six-unit scope.
  Identical predictor tensors were paired between the two auxiliary models;
  the no-aux baseline has a different input dimension.
- Lorenz: `source_data/phase9_lorenz_formal_confirmation/` includes the frozen
  configuration, five data-seed summaries, and decision report. Two training
  seeds are averaged before sample standard deviations across five data seeds.
  Entropy 0.956+/-0.003, temporal tail 0.182+/-0.018, H64/H128 mass ratio
  minimum approximately 1, and nominal horizon difference zero all pass their
  separate conditions in 5/5 seeds. The time grid is linspace(0,150,1500),
  retaining 500 observations after 1,000 burn-in points. Ground truth denotes
  ODE coupling support, not a proved exact order-one GC graph after sampling
  and observation noise.

## Existing derivative checks

| Included test/source | Evidence and scope |
| --- | --- |
| `code/tests/test_jacobian_coverage_audit.py` | Double-precision chain decomposition, coordinate counterexample, and finite-support formulas. |
| `code/tests/test_phase8_coverage.py`, `code/src/phase8_coverage.py` | Implemented raw-chain finite differences and direct-plus-indirect decomposition, estimator and second-order checks. |
| `code/tests/test_p0_repaired_istf.py` | Actual depthwise finite memory, depthwise/Mamba raw-chain finite differences, and disconnected-input negative control. |
| `code/tests/test_phase9_audit_generalization.py` | TCN future-perturbation causality, output-projection gradients, and sampled audit execution. This is not an active h=7 boundary test. |

`REPRODUCE.md` gives independent validation commands. Source hashes identify
the supplied computation; diagnostic support checks do not replace the
assumptions needed for a statistical or causal interpretation.
