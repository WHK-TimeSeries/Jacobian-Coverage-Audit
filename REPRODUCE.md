# Reproduction Guide

## 1. CPU installation

Use 64-bit Python 3.10. The clean Windows validation used Python 3.10.19;
the archived Linux GPU runs used 3.10.20. Internet access is needed for the
initial PyPI/PyTorch installation, not for the quick checks below. Do not
install into another project's environment. No Git executable, checkout,
release token, adjacent JRNGC folder, or host PYTHONPATH is required.

Open a terminal in the extracted `KBS_Supplementary_Code` directory. Create
the environments and all outputs OUTSIDE it. In PowerShell:

```powershell
python -m venv ../kbs-cpu
../kbs-cpu/Scripts/python.exe -m pip install pip==25.2
../kbs-cpu/Scripts/python.exe -m pip install -r environment/requirements-cpu.lock
../kbs-cpu/Scripts/python.exe -m pip check
```

On Linux/macOS, the corresponding interpreter path is `../kbs-cpu/bin/python`.
The CPU lock is validated on Windows x86-64 only; availability on other
platforms is not a tested claim. It explicitly installs `torch==2.11.0+cpu`,
not the original CUDA wheel. The complete captured formal environments remain
in `environment/phase9_*`; CPU validation does not promise bitwise equality
with a CUDA training run.

## 2. Documented quick checks

Run these commands with the CPU interpreter. Each output must be a NEW
directory. Commands locate their inputs relative to `reproduce.py`, so they
also work when invoked by absolute path from an unrelated working directory.

```powershell
../kbs-cpu/Scripts/python.exe -B reproduce.py verify --output ../kbs-out-verify
../kbs-cpu/Scripts/python.exe -B reproduce.py tests --output ../kbs-out-tests
../kbs-cpu/Scripts/python.exe -B reproduce.py example --output ../kbs-out-example
../kbs-cpu/Scripts/python.exe -B reproduce.py tables --output ../kbs-out-tables
../kbs-cpu/Scripts/python.exe -B reproduce.py experiment --protocol lorenz --output ../kbs-plan-lorenz
../kbs-cpu/Scripts/python.exe -B reproduce.py experiment --protocol stageb --output ../kbs-plan-stageb
```

- `verify`: checks the distributed files, required dependencies, and unexpected
  Python modules. A changed/missing source file is a hard failure.
- `tests`: runs the supplied Phase 9 suites and portable-entry tests, including
  native comparator parity, chain decomposition, finite differences, frozen
  matrix validation, and checkpoint-selection parity. JUnit XML is saved.
- `example`: trains baseline, Mamba-concat, and TCN-concat on a small synthetic
  Lorenz fixture for 20 iterations, twice each. Checks CPU determinism and
  baseline partial/total agreement. These six toy runs are not paper evidence.
- `tables`: parses and exports the distributed CSV inputs to one JSON index.
  This is source-table reconstruction, not independent reaggregation from
  unbundled original checkpoints.
- `experiment` without `--execute`: validates and lists the frozen 20-row
  Lorenz or 36-row Stage B matrix. It does not train or load external arrays.

Every command writes `validation.json` and `command.log`, exits nonzero on
failure, and verifies the archive again after successful execution. The
archive is never used as an output/cache directory.

## 3. Numerical figures

Use a separate environment to retain the accepted plotting versions:

```powershell
python -m venv ../kbs-figures
../kbs-figures/Scripts/python.exe -m pip install pip==25.2
../kbs-figures/Scripts/python.exe -m pip install -r environment/requirements-figures.lock
../kbs-figures/Scripts/python.exe -m pip check
../kbs-figures/Scripts/python.exe -B reproduce.py figures --output ../kbs-out-figures
```

The command copies Figure 3, Figure 4, Figure 5, and Figure D.1 directories
to the output, removes only the copied exports, and executes their unchanged
Python plotting scripts. It checks new PDF/PNG/SVG outputs, nonblank pixels,
and unchanged input CSVs. Figures 1 and 2 are conceptual; their editable sources
are included but are not numerical regeneration targets. The README figure map
relates the stable source-directory names to current manuscript numbers. Font substitution and
PDF metadata can differ across operating systems; numerical source identity
is checked separately from byte-for-byte PDF identity.

## 4. Optional, expensive independent training

The portable runner reuses the frozen model, generator/data loader, training
function, checkpoint-selection rule, intervention functions, and Jacobian
evaluators. It does NOT forge the original release approval or weaken the
original formal entry points. New outputs are always labelled
`formal_result=false` and `independent_reproduction_not_original_formal_result`.

After inspecting a saved plan, execute one selected, 1-based matrix row:

```powershell
../kbs-cpu/Scripts/python.exe -B reproduce.py experiment --protocol lorenz --execute --row 1 --output ../kbs-lorenz-row1
../kbs-cpu/Scripts/python.exe -B reproduce.py experiment --protocol stageb --execute --row 1 --data-root ../external-data --output ../kbs-stageb-row1
```

These commands have NO reduced-budget default: they use the frozen row's
training budget and checkpoint policy, and can be slow on CPU. They were
dry-validated but NOT run as full experiments during this packaging task.
Repeat with new output directories for the other matrix rows as needed.
Stage B requires the exact external NPZ files described in
`THIRD_PARTY_DATA.md`; hashes are checked before loading arrays. A CUDA rerun
requires a separately prepared environment consistent with the captured
formal stack and explicit `--device cuda`; no silent CPU/GPU fallback exists.
The CPU lock must not be used as a CUDA environment.

## 5. Historical pipelines and limits

The scientific computation in `code/` is retained from the research snapshot
in `PROVENANCE_COMMITS.txt`. Publication docstrings and source documentation
may differ. `PUBLICATION_SOURCE_MAP.json` records both file fingerprints;
frozen model configurations, matrices and numerical source tables are unchanged.
Original Phase 8/9 formal release entry points,
including their Git/token guards, remain archival execution records. The
portable entry covers the current Stage B/Lorenz audit workflows; it is not
a replacement formal launcher for every historical Phase 8 repair experiment.
The repair figures/tables can be regenerated from their included frozen data.
Original checkpoints, full run trees, and third-party datasets are not in
this review ZIP. Full historical retraining and all-platform/GPU validation
are outside this code-packaging acceptance check.
