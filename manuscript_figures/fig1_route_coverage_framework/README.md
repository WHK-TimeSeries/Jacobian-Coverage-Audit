# Figure 1: Partial and Total Jacobian Graph Readouts

The editable diagram is `fig1_route_coverage_framework.drawio`. Matching PDF,
SVG and PNG exports show the same diagram. The PDF is vector artwork, 180 mm wide.

The forward predictor receives concatenated input [X; C]. The partial
derivative is a readout, not a forward-processing layer. Matrix rows are
targets and columns are sources. A common off-diagonal threshold of 0.5
gives the illustrated graphs; signed contributions are added before taking
magnitudes. Cancellation removes edge 2 -> 1; the auxiliary route adds
edge 3 -> 2. Edge 1 -> 3 remains.

`data/schematic_jacobians.json` records analytical examples, not experimental
data or graph ground truth. Edit the draw.io file directly and export the
whole drawing as PDF/SVG/PNG, keeping all arrows visible.
