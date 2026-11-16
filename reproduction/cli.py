from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .integrity import bootstrap, new_output, sha256, verify_archive, write_json


def figures(root: Path, output: Path) -> dict:
    rendered = []
    for source in sorted((root / "manuscript_figures").iterdir()):
        script = source / "scripts/plot_figure.py"
        if not script.is_file():
            continue
        copied = output / source.name
        shutil.copytree(source, copied)
        # Remove copied exports in the NEW output only, so existence checks
        # cannot mistake an archived PDF for a successful regeneration.
        files = {ext: copied / f"{source.name}{ext}" for ext in (".png", ".pdf", ".svg")}
        for path in files.values():
            if path.exists():
                path.unlink()
        result = subprocess.run([sys.executable, "-B", str(copied / "scripts/plot_figure.py")],
                                cwd=output, text=True, capture_output=True)
        (copied / "plot.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Figure failed: {source.name}; see plot.log")
        if not all(p.is_file() and p.stat().st_size > 1000 for p in files.values()):
            raise ValueError(f"Incomplete figure regeneration: {source.name}")
        import numpy as np
        from PIL import Image
        pixels = np.asarray(Image.open(files[".png"]).convert("RGB"))
        if float(pixels.std()) < 1 or min(pixels.shape[:2]) < 100:
            raise ValueError(f"Blank/invalid rendered image: {source.name}")
        data_hashes = {p.relative_to(source).as_posix(): sha256(p) for p in (source / "data").rglob("*") if p.is_file()}
        if any(sha256(copied / rel) != digest for rel, digest in data_hashes.items()):
            raise ValueError(f"Plot script changed frozen data: {source.name}")
        rendered.append({"figure": source.name, "outputs": {ext: sha256(p) for ext, p in files.items()},
                         "png_shape": list(pixels.shape), "data_unchanged": True})
    if len(rendered) != 4:
        raise ValueError(f"Expected four numerical figures, found {len(rendered)}")
    return {"passed": True, "figures": rendered, "conceptual_figure_1": "preserved; no numerical regeneration"}


def tables(root: Path, output: Path) -> dict:
    """Validate and collect distributed table inputs without inventing raw runs."""
    records = []
    for base in (root / "source_data", root / "manuscript_figures"):
        for source in sorted(base.rglob("*.csv")):
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                if not reader.fieldnames or any(None in row for row in rows):
                    raise ValueError(f"Malformed CSV: {source}")
            records.append({"path": source.relative_to(root).as_posix(), "sha256": sha256(source),
                            "columns": reader.fieldnames, "rows": rows})
    if not records:
        raise ValueError("No table inputs found")
    write_json(output / "frozen_table_inputs.json", records)
    return {"passed": True, "csv_count": len(records), "rows": sum(len(r["rows"]) for r in records),
            "role": "frozen source-table reconstruction, not reaggregation from unbundled checkpoints"}


def tests(root: Path, output: Path) -> dict:
    import pytest
    import torch
    torch.set_num_threads(1)
    paths = [root / "code/tests/test_phase9_audit_generalization.py",
             root / "code/tests/test_phase9_lorenz_901_confirmation.py",
             root / "reproduction/test_portable.py"]
    result = pytest.main([*map(str, paths), "-q", "-p", "no:cacheprovider",
                          "--basetemp", str(output / "pytest-temp"),
                          "--junitxml", str(output / "junit.xml")])
    if result:
        raise RuntimeError(f"pytest failed with exit code {result}")
    from xml.etree import ElementTree
    xml = ElementTree.parse(output / "junit.xml")
    count = sum(int(x.get("tests", "0")) for x in xml.findall("testsuite"))
    return {"passed": True, "tests": count, "junit": "junit.xml"}


def main(root: Path) -> int:
    parser = argparse.ArgumentParser(description="Independent supplementary-code reproduction (no Git required)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "tests", "example", "figures", "tables", "experiment", "prepare-netsim"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True, help="New directory outside the archive")
        if name == "prepare-netsim":
            command.add_argument("--source", type=Path, required=True, help="Downloaded NetSim sim3.mat")
        if name == "experiment":
            command.add_argument("--protocol", choices=("lorenz", "stageb"), required=True)
            command.add_argument("--execute", action="store_true", help="Opt in to expensive independent training")
            command.add_argument("--row", type=int, help="1-based row from the frozen plan")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        before = verify_archive(root)
        output = new_output(root, args.output)
    except (ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    bootstrap(root, output, getattr(args, "device", "cpu"))
    report = {"command": args.command, "passed": False, "archive_before": before,
              "formal_result": False, "scientific_evidence": False}
    with (output / "command.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                if args.command == "verify":
                    result = before
                elif args.command == "tests":
                    result = tests(root, output)
                elif args.command == "example":
                    from .workflows import smoke
                    result = smoke(root, output)
                elif args.command == "experiment":
                    from .workflows import experiment
                    result = experiment(root, output, args)
                elif args.command == "prepare-netsim":
                    from .datasets import prepare_netsim
                    result = prepare_netsim(root, output, args.source)
                elif args.command == "figures":
                    result = figures(root, output)
                else:
                    result = tables(root, output)
                after = verify_archive(root)
                if before != after:
                    raise ValueError("Archive changed during reproduction")
                report.update({"passed": True, "result": result, "archive_after": after})
        except Exception as exc:
            traceback.print_exc(file=log)
            report["error"] = f"{type(exc).__name__}: {exc}"
    report["wall_seconds"] = time.perf_counter() - started
    write_json(output / "validation.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.command}; {output / 'validation.json'}")
    return 0 if report["passed"] else 1
