# Vendored JRNGC metric dependency

Only `tgc/metrics/__init__.py` and `tgc/metrics/causal.py` are bundled.
They are byte-identical to the local JRNGC snapshot at commit
`974f242f21f055f7b9b535664c7c4f7a8d821994`; these files were clean in that
checkout when copied. This identifies the supplied snapshot, not an
unverified claim of identity with every historical server installation.

Original work: Wanqi Zhou et al., *Jacobian Regularizer-based Neural Granger
Causality*, ICML 2024, https://proceedings.mlr.press/v235/zhou24a.html.
Copyright (c) 2024 Wanqi_Zhou. See the unmodified MIT `LICENSE` alongside
this notice. The metric implementation is not replaced by a similarly named
PyPI package or by a different AUROC/AUPRC routine.

| File | checksum |
| --- | --- |
| tgc/metrics/__init__.py | 1fdb91bbf1b0c27dd2f1638bc9ad8dc9a727bb58ddd64f4d40111dbd23299eb8 |
| tgc/metrics/causal.py | 4b079e1b01acc14b25f42d428b6fa6d9d1f328d1ea2cf503b23415eec75c8786 |
| LICENSE | 76f31ced343f144ed777606bb9a41f79a79dc3b07c136cc03124e710ab77e367 |
