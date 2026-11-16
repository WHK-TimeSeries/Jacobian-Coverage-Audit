# Figure 5: graph--prediction frontier

Rebuild from the frozen trade-off CSV with:

```powershell
python scripts\plot_figure.py
```

The Python script writes vector PDF/SVG, a 600-dpi PNG, numerical validation,
and checksum manifests.

The categorical repair gates are not embedded in the figure. Their frozen CSV
and English/Chinese LaTeX table sources are maintained under
`source_data/phase9_repair_gate_table/`.
