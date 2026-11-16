from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIGURES = [
    ("Figure 1", "fig1_route_coverage_framework", "drawio"),
    ("Figure 2", "fig2_coverage_audit_framework", "drawio"),
    ("Figure 3", "fig2_controlled_decoupling", "python"),
    ("Figure 4", "fig3_heldout_audit_generality", "python"),
    ("Figure 5", "fig4_graph_prediction_frontier", "python"),
    ("Figure D.1", "figE1_score_semantics", "python"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records: list[dict[str, object]] = []
    for manuscript_label, directory_name, backend in FIGURES:
        directory = ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        pdf = directory / f"{directory_name}.pdf"
        svg = directory / f"{directory_name}.svg"
        png = directory / f"{directory_name}.png"
        manifest = directory / "provenance" / "CHECKSUMS.txt"
        for required in (pdf, svg, png, manifest):
            if not required.is_file():
                raise FileNotFoundError(required)
        source = (
            directory / f"{directory_name}.drawio"
            if backend == "drawio"
            else directory / "scripts" / "plot_figure.py"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            {
                "manuscript_label": manuscript_label,
                "figure": directory_name,
                "backend": "draw.io" if backend == "drawio" else "Python/matplotlib",
                "artifact_count": sum(1 for path in directory.rglob("*") if path.is_file()),
                "source": source.relative_to(directory).as_posix(),
                "source_sha256": sha256(source),
                "pdf": pdf.name,
                "pdf_sha256": sha256(pdf),
                "svg": svg.name,
                "svg_sha256": sha256(svg),
                "png": png.name,
                "png_sha256": sha256(png),
                "manifest": manifest.relative_to(directory).as_posix(),
                "manifest_sha256": sha256(manifest),
            }
        )
    with (ROOT / "FIGURE_ASSET_INDEX.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
