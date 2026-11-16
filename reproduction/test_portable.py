from pathlib import Path
import os
import shutil

import numpy as np
import pytest
import torch

from reproduction.integrity import new_output, verify_archive
from reproduction.workflows import protocol_plan

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_dependency_cannot_resolve_to_host_checkout():
    import tgc.metrics
    source = Path(tgc.metrics.__file__).resolve()
    assert source.is_relative_to(ROOT / "third_party/JRNGC")
    assert os.environ["JRNGC_PATH"] == str(ROOT / "third_party/JRNGC")


def test_verification_does_not_require_git():
    assert verify_archive(ROOT)["git_required"] is False
    assert not (ROOT / ".git").exists()


def test_missing_and_modified_source_are_rejected(tmp_path):
    fixture = tmp_path / "archive"
    shutil.copytree(ROOT, fixture)
    target = fixture / "code/src/phase8_coverage.py"
    target.write_bytes(target.read_bytes() + b"\n# corruption fixture\n")
    with pytest.raises(ValueError, match="File checksum mismatch"):
        verify_archive(fixture)
    target.unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_archive(fixture)


@pytest.mark.parametrize("name", ["injected.py", "injected.pyc"])
def test_injected_python_source_is_rejected(tmp_path, name):
    fixture = tmp_path / "archive"
    shutil.copytree(ROOT, fixture)
    (fixture / "code/src" / name).write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="Unlisted"):
        verify_archive(fixture)


def test_output_inside_archive_or_existing_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        new_output(ROOT, ROOT / "outputs")
    with pytest.raises(ValueError, match="exists"):
        new_output(ROOT, tmp_path)


@pytest.mark.parametrize("protocol,count", [("lorenz", 20), ("stageb", 36)])
def test_frozen_independent_plans(protocol, count):
    path, config, rows = protocol_plan(ROOT, protocol)
    assert len(rows) == count
    assert path.is_file()
    assert config["model"]["jacobian_lam"] == .01


def test_adapter_native_loss_and_gradient_parity():
    from phase8_coverage import comparator_parity_audit
    report = comparator_parity_audit()
    assert report["passed"] is True


def test_total_chain_finite_difference_and_decomposition():
    from phase8_coverage import finite_difference_total_raw_chain_audit, direct_indirect_chain_decomposition_audit
    assert finite_difference_total_raw_chain_audit()["passed"] is True
    assert direct_indirect_chain_decomposition_audit()["passed"] is True


def test_short_training_matches_frozen_checkpoint_policy(tmp_path):
    from phase8_coverage import Phase8ModelConfig, make_legacy_baseline
    from phase9_audit_stageb import train_legacy_with_metadata
    cfg = Phase8ModelConfig(d=3, lag=2, layers=1, hidden=6)
    torch.manual_seed(12345)
    model = make_legacy_baseline(cfg).model
    twin = make_legacy_baseline(cfg).model
    twin.load_state_dict(model.state_dict())
    x = np.random.default_rng(12345).normal(size=(3, 16)).astype(np.float32)
    report = train_legacy_with_metadata(model, x, max_iter=6, learning_rate=.001,
        weight_decay=0., gradient_clip_norm=1., check_every=2, lookback=10)
    optimizer = torch.optim.Adam(twin.parameters(), lr=.001, weight_decay=0.)
    best = float("inf")
    best_state = None
    import copy
    for iteration in range(6):
        optimizer.zero_grad()
        loss = twin.compute_loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(twin.parameters(), 1.)
        optimizer.step()
        if iteration % 2 == 0 and float(loss.detach()) < best:
            best = float(loss.detach())
            best_state = copy.deepcopy(twin.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, best_state[key], atol=1e-7, rtol=1e-6)
    assert report["best_total_regularized_objective"] == best


def test_netsim_conversion_preserves_orientation_and_refuses_overwrite(tmp_path):
    from reproduction.datasets import export_subject
    graph = np.zeros((2, 3, 3))
    graph[1, 0, 2] = 1
    data = {"Ntimepoints": 5, "Nnodes": 3,
            "ts": np.arange(30).reshape(10, 3), "net": graph}
    target = tmp_path / "subject.npz"
    export_subject(data, 1, target)
    with np.load(target, allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["X_np"], data["ts"][5:10].T)
        assert exported["Gref"][2, 0] == 1
        assert exported["Gref"].sum() == 1
    with pytest.raises(FileExistsError):
        export_subject(data, 1, target)


@pytest.mark.parametrize("method", ["baseline", "mamba_concat"])
@pytest.mark.parametrize("reseed", [False, True])
def test_portable_training_matches_direct_frozen_apis(tmp_path, method, reseed):
    from phase8_coverage import Phase8ModelConfig
    from phase9_audit_stageb import make_adapter, train_legacy_with_metadata
    from phase9_lorenz_strong_operating_preflight import configure_determinism
    from reproduction.workflows import run_case
    cfg = Phase8ModelConfig(d=3, lag=2, layers=1, hidden=6, d_cond=2,
                            d_state=2, d_conv=2, expand=1)
    x = np.random.default_rng(12345).normal(size=(3, 14)).astype(np.float32)
    seeds = dict(master_seed=12346, predictor_seed=12346, preprocessor_seed=1012346,
                 score_window_seed=12347, perturbation_seed=12348)
    config = {"training": {"learning_rate": .001, "weight_decay": 0.,
                            "gradient_clip_norm": 1., "check_every": 2, "lookback": 10},
              "evaluation": {"primary_horizon": 6, "horizon_sensitivity": [6],
                              "common_target_minimum_horizon": 6, "audit_window_count": 2}}
    _, _, _, state = run_case(x, None, cfg, config, seeds, method, 4,
                             tmp_path / "portable", "cpu", reseed_after_initialization=reseed)
    configure_determinism(seeds["master_seed"], torch.device("cpu"))
    direct = make_adapter(method, cfg, predictor_seed=seeds["predictor_seed"],
                          preprocessor_seed=None if method == "baseline" else seeds["preprocessor_seed"])
    if reseed:
        torch.manual_seed(seeds["master_seed"])
        np.random.seed(seeds["master_seed"] % (2**32 - 1))
    train_legacy_with_metadata(direct.model, x, max_iter=4, **config["training"])
    for key, value in direct.model.state_dict().items():
        np.testing.assert_array_equal(state[key], value.detach().numpy())
