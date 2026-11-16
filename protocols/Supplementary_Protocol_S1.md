# Supplementary Protocol S1: Coordinate-Preserving Filter Control

This protocol accompanies the independent control study in Appendix C. It preserves the experimental settings and decision criteria of that study; no new experiment or reanalysis is reported here.

## Design and Analysis Unit

The four methods were baseline JRNGC, a trainable coordinate-preserving depthwise filter (CP-depthwise), a fixed three-tap finite-impulse-response control (FixedFIR3), and fixed exponential smoothing (FixedEMA). The four D2 cells were Stat+Linear, Stat+Nonlinear, NS+Linear, and NS+Nonlinear.

Each series had dimension d=10, length T=600, and nominal lag K=3. Data seeds 1, 2, and 3 were paired with training seeds 0 and 1, giving 4 cells x 4 methods x 3 data seeds x 2 training seeds = 96 runs. The primary checkpoint was fixed at iteration 500. The statistical unit was the data seed after averaging its two training seeds.

Baseline JRNGC, CP-depthwise, and FixedFIR3 used attribution horizon H=32 and nominal-lag raw-chain graph scores over K=3. FixedEMA used H=64 for both training attribution and evaluation, and its graph scores were evaluated as a full-H reference, not a nominal-lag method.

## Generator

Each off-diagonal edge was included with probability 0.2 and assigned one uniformly chosen lag. Signed coefficient magnitudes were uniform on [0.12, 0.40]. Non-stationary coefficients added support-masked Gaussian random-walk increments of scale 0.08/sqrt(600), smoothed over the trailing indices max(0,t-20):t.

For p_t = sum_{k=1}^{3} A_{t,k} x_{t-k}, the linear cells used p_t and the nonlinear cells used the coordinate-wise link

```text
g(p_t) = 0.5 p_t + 0.5 (0.075) tanh(p_t / 0.075).
```

Independent innovations had standard deviation 0.15. The series were not standardized. The graph labels were the declared nonzero coefficient support.

## Training and Scoring

All four models used one residual block of width 16, Adam with learning rate 0.001, global gradient clipping at 1, and float32 training. Raw-target prediction MSE used the same target indices u=64,...,599 for every method.

The sampled raw-chain Jacobian penalty used two windows and two output rows per step. Its normalization used the actual sampled counts, source dimension d, and nominal lag K, with lambda_J=0.01; increasing the attribution horizon did not change that denominator.

CP-depthwise used a zero-initialized causal depthwise kernel of width 3, residual gain 0.1, and identity-penalty coefficient 0.05. FixedEMA used alpha=0.9 and initial state z_0=x_0. Exact graph scoring used all common windows with float64 accumulation.

## Pre-Specified Decision Criteria

The CP-versus-baseline criterion required at least two qualifying cells, including a non-stationary cell. Each qualifying cell had to meet all four conditions:

- Mean Delta AUROC >= 0.03.
- Positive Delta AUROC in at least two of three data seeds.
- Mean Delta AUPRC >= -0.02.
- Mean Delta MCC >= -0.05.

In at least one of those qualifying cells, CP versus FixedFIR3 additionally had to meet all four conditions:

- Mean Delta AUROC >= 0.02.
- Positive Delta AUROC in at least two of three data seeds.
- Mean Delta AUPRC >= -0.02.
- Mean Delta MCC >= -0.05.

The EMA dominance stopping criterion required EMA minus CP mean Delta AUROC >= 0 in all four cells and >= 0.02 in at least two cells, together with mean Delta AUPRC >= -0.02 and mean Delta MCC >= -0.05 in every cell. EMA was assessed on its declared full-H reference track.

## Recorded Outcome

All declared semantic checks passed. The CP-versus-baseline performance criterion and the CP-versus-FixedFIR3 criterion were not met. The EMA dominance stopping criterion did not trigger. FixedFIR3 achieved higher cell-level mean AUROC than CP-depthwise in all four cells, although not in every individual data seed. Appendix Table C.1 reports the unchanged results.

These outcomes distinguish validity of the graph-score definition from an empirical method advantage. The small stationary-cell differences do not establish an advantage for learned depthwise filtering.
