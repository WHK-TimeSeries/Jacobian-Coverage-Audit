# Figure 2: A Worked Jacobian Coverage Audit

The editable diagram is `fig2_coverage_audit_framework.drawio`. Matching PDF,
SVG and PNG exports show the same diagram. The 180 mm PDF retains vector text
and diagram elements, with one embedded waveform thumbnail (873 x 1281 pixels).

The x-only concat example has aligned score and penalty masks, but both omit
the auxiliary route. Raw-score columns retain source identity; omitted
auxiliary temporal support remains unassessed. Clean and masked-C inputs use
the same predictor and fixed raw target. MSE excludes regularization penalties.

`data/illustrative_matrices.json` contains conceptual inputs, not measured
results. Edit the draw.io source directly and export the whole diagram.
The manuscript uses this illustration in Section 4.
