[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$scripts = @(
    "fig2_controlled_decoupling\scripts\plot_figure.py",
    "fig3_heldout_audit_generality\scripts\plot_figure.py",
    "fig4_graph_prediction_frontier\scripts\plot_figure.py",
    "figE1_score_semantics\scripts\plot_figure.py"
)

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $Python"
}

foreach ($relativeScript in $scripts) {
    $script = Join-Path $root $relativeScript
    & $Python $script
    if ($LASTEXITCODE -ne 0) {
        throw "Figure build failed: $relativeScript"
    }
}

& $Python (Join-Path $root "update_asset_index.py")
if ($LASTEXITCODE -ne 0) {
    throw "Figure asset-index update failed."
}

Write-Output "Rebuilt four Python numerical figures and refreshed FIGURE_ASSET_INDEX.json."
