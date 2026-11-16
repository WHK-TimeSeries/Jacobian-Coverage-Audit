"""Byte-stable NetSim export using the original Stage B conversion recipe."""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from .integrity import sha256


def export_subject(data: dict, subject: int, destination: Path) -> None:
    import numpy as np
    timepoints = int(np.asarray(data["Ntimepoints"]).item())
    nodes = int(np.asarray(data["Nnodes"]).item())
    start = subject * timepoints
    x = np.asarray(data["ts"][start:start + timepoints], dtype=np.float64).T
    graph = (np.asarray(data["net"][subject]) != 0).astype(np.int16).T
    if x.shape != (nodes, timepoints) or graph.shape != (nodes, nodes):
        raise ValueError("Unexpected NetSim dimensions")
    arrays = (("X_np.npy", x), ("Gref.npy", graph),
              ("n.npy", np.int64(nodes)), ("T.npy", np.int64(timepoints)))
    with zipfile.ZipFile(destination, mode="x", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, value in arrays:
            payload = io.BytesIO()
            np.save(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())


def prepare_netsim(root: Path, output: Path, source: Path) -> dict:
    import scipy.io
    config = json.loads((root / "code/configs/phase9_audit_stageb_v1.json").read_text(encoding="utf-8-sig"))
    data = scipy.io.loadmat(source)
    (output / "netsim").mkdir()
    files = []
    for name, spec in config["datasets"].items():
        if spec["kind"] != "netsim":
            continue
        subject = int(Path(spec["relative_path"]).stem.rsplit("_", 1)[1])
        target = output / spec["relative_path"]
        export_subject(data, subject, target)
        actual = sha256(target)
        if actual != spec["sha256"]:
            raise ValueError(f"NetSim source/export does not match frozen bytes: {name}")
        files.append({"file": spec["relative_path"], "sha256": actual})
    return {"passed": True, "source_sha256": sha256(source), "files": files,
            "operation": "format conversion only; no training or graph evaluation"}
