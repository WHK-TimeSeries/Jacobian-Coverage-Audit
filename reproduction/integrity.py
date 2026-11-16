from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(root: Path) -> dict:
    """Validate distributed bytes, not an unrelated enclosing Git repository."""
    manifest = root / "CHECKSUMS.txt"
    entries = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        digest, relative = line.split("  ", 1)
        relative = relative.replace("\\", "/")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or ":" in relative:
            raise ValueError(f"Unsafe manifest path: {relative}")
        if relative in entries or len(digest) != 64:
            raise ValueError(f"Invalid/duplicate manifest entry: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping archive file: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"File checksum mismatch: {relative}")
        entries[relative] = digest
    required = ("reproduce.py", "reproduction/cli.py", "code/src/phase8_coverage.py",
                "third_party/JRNGC/tgc/metrics/causal.py", "RELEASE.json")
    if not all(name in entries for name in required):
        raise ValueError("Incomplete archive manifest")
    # -B prevents writes, not reads of existing bytecode. Reject unlisted
    # source, bytecode, and native extension files in the distributed code.
    for directory in ("code", "third_party", "reproduction"):
        for path in (root / directory).rglob("*"):
            if path.suffix.lower() in {".py", ".pyc", ".pyo", ".pyd", ".so"} and path.relative_to(root).as_posix() not in entries:
                raise ValueError(f"Unlisted Python source: {path.relative_to(root)}")
    release = json.loads((root / "RELEASE.json").read_text(encoding="utf-8-sig"))
    return {"passed": True, "integrity_mode": "archive_sha256_manifest",
            "files_verified": len(entries), "source_manifest_sha256": sha256(manifest),
            "research_snapshot_commit": release["research_snapshot_commit"],
            "packaging_commit": release["packaging_commit"],
            "git_required": False, "formal_release_authorization": False}


def new_output(root: Path, requested: Path) -> Path:
    output = requested.resolve()
    if output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output):
        raise ValueError("Output must be outside the extracted archive and not its ancestor")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    return output


def bootstrap(root: Path, output: Path, device: str = "cpu") -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Always select the bundled MIT dependency, including with a stale host env.
    os.environ["JRNGC_PATH"] = str(root / "third_party" / "JRNGC")
    paths = [root / "third_party" / "JRNGC", root / "code" / "src",
             root / "code", root / "code" / "experiments", root]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))


def write_json(path: Path, payload: object) -> None:
    def numeric_default(value):
        # Frozen scientific APIs sometimes return NumPy scalar/array objects.
        if type(value).__module__.startswith("numpy"):
            return value.tolist()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False,
                               default=numeric_default) + "\n",
                    encoding="utf-8")
