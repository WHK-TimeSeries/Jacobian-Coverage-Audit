# Manuscript figures

Each figure has its own source, data and exports.

| Figure | Source |
| --- | --- |
| Figure 1: Jacobian graph readouts | Editable draw.io schematic |
| Figure 2: coverage audit framework | Editable draw.io schematic |
| Figure 3: controlled decoupling | Python/matplotlib and frozen CSVs |
| Figure 4: held-out audit validation | Python/matplotlib and frozen CSV/JSON |
| Figure 5: graph-prediction trade-off | Python/matplotlib and frozen CSV |
| Figure D.1: score semantics | Python/matplotlib and frozen CSV |

Use the archive-root `REPRODUCE.md` for the independent numerical-figure
command and pinned plotting environment. Individual numerical directories
also contain `scripts/plot_figure.py` and `requirements.txt`.

Figures 1 and 2 are analytical schematics, not experimental evidence. Their editable
sources and ordinary draw.io export helpers are supplied separately from the
numerical plotting scripts.

Source-directory identifiers remain stable. The archive-root README and
`FIGURE_ASSET_INDEX.json` map them to the current manuscript labels.
Graph definitions, analysis units and uncertainty conventions are given in
`SCORING_PROTOCOL.md`, the figure captions and data files. Manifests apply to
the distributed files; edit a copy outside the verified archive.
