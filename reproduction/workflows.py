from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import time

from .integrity import sha256, write_json


def environment(device: str) -> dict:
    import numpy as np
    import scipy
    import sklearn
    import torch
    import tgc.metrics
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "torch_cuda_build": torch.version.cuda, "device": device,
            "tgc_metrics_source": str(Path(tgc.metrics.__file__).resolve()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def run_case(x, graph, cfg, config, seeds, method, iterations, output, device,
             *, reseed_after_initialization=True):
    """Reuse frozen predictor/trainer/evaluator APIs; only write reviewer metadata."""
    import numpy as np
    import torch
    from phase8_coverage import fixed_target_concat_interventions
    from phase9_audit_generalization import (
        deterministic_audit_targets, sampled_raw_chain_audit,
        fixed_target_baseline_interventions, condition_coordinate_mixing_audit,
    )
    from phase9_audit_stageb import make_adapter, train_legacy_with_metadata, horizon_summary, all_finite
    from phase9_lorenz_strong_operating_preflight import configure_determinism

    output.mkdir(parents=True)
    torch.set_num_threads(1)
    target_device = torch.device(device)
    deterministic = configure_determinism(seeds["master_seed"], target_device)
    adapter = make_adapter(method, cfg, predictor_seed=seeds["predictor_seed"],
                           preprocessor_seed=None if method == "baseline" else seeds["preprocessor_seed"])
    adapter.model.to(target_device)
    # Preserve the separate frozen RNG policies: Stage B reseeds here,
    # whereas Lorenz retains the RNG state after paired initialization.
    if reseed_after_initialization:
        torch.manual_seed(seeds["master_seed"])
        np.random.seed(seeds["master_seed"] % (2**32 - 1))
        if device != "cpu":
            torch.cuda.manual_seed_all(seeds["master_seed"])
    started = time.perf_counter()
    train = config["training"]
    training = train_legacy_with_metadata(
        adapter.model, x, max_iter=iterations,
        learning_rate=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]),
        gradient_clip_norm=float(train["gradient_clip_norm"]),
        check_every=int(train["check_every"]), lookback=int(train["lookback"]),
    )
    ev = config["evaluation"]
    indices = deterministic_audit_targets(
        T=x.shape[1], lag=cfg.lag, attribution_horizon=ev["common_target_minimum_horizon"],
        count=ev["audit_window_count"], seed=seeds["score_window_seed"])
    horizons = [ev["primary_horizon"]] if method == "baseline" else ev["horizon_sensitivity"]
    audits = {h: sampled_raw_chain_audit(adapter, x, target_indices=indices,
                                         attribution_horizon=h, graph=graph) for h in horizons}
    primary = audits[ev["primary_horizon"]]
    intervention_fn = fixed_target_baseline_interventions if method == "baseline" else fixed_target_concat_interventions
    interventions = intervention_fn(adapter, x, perturbation_seed=seeds["perturbation_seed"])
    mixing = None if method == "baseline" else condition_coordinate_mixing_audit(
        adapter, x, target_indices=indices, attribution_horizon=ev["primary_horizon"])
    horizon = horizon_summary(audits) if set(audits) == {32, 64, 128} else None
    components = {k: float(v.detach().cpu()) for k, v in adapter.loss_components(x).items()}
    report = {"result_role": "independent_reproduction_not_original_formal_result",
              "formal_result": False, "manuscript_evidence": False, "method": method,
              "iterations": iterations, "model_config": asdict(cfg), "seeds": seeds,
              "target_indices": indices.tolist(), "training": training,
              "loss_components": components, "primary": primary.summary(),
              "interventions": interventions, "coordinate_mixing": mixing,
              "horizon_sensitivity": horizon, "determinism": deterministic,
              "reseed_after_initialization": reseed_after_initialization,
              "wall_seconds": time.perf_counter() - started}
    if not all_finite(report):
        raise ValueError("Nonfinite reproduction output")
    for h, audit in audits.items():
        np.savez_compressed(output / f"attribution_H{h}.npz", **audit.arrays())
    with torch.no_grad():
        predictions = adapter.predictions(x).cpu().numpy()
    np.save(output / "predictions.npy", predictions)
    torch.save({"model_state": adapter.model.state_dict(), "formal_result": False,
                "model_config": asdict(cfg)}, output / "checkpoint.pt")
    report["no_nan_inf"] = True
    write_json(output / "run.json", report)
    state = {k: v.detach().cpu().numpy().copy() for k, v in adapter.model.state_dict().items()}
    return report, primary, predictions, state


def smoke(root: Path, output: Path) -> dict:
    import numpy as np
    from phase8_coverage import Phase8ModelConfig
    from lorenz96_frozen import generate_lorenz96

    cfg = Phase8ModelConfig(d=6, lag=2, layers=1, hidden=8, d_cond=2,
                            d_state=2, d_conv=2, expand=1)
    x, _, graph = generate_lorenz96(d=6, t=36, t_eval=0, forcing=40., seed=12345,
                                    delta_t=.1, observation_noise_sd=.1, burn_in=60)
    config = {"training": {"learning_rate": .001, "weight_decay": 0.,
                            "gradient_clip_norm": 1., "check_every": 5, "lookback": 10},
              "evaluation": {"primary_horizon": 8, "horizon_sensitivity": [8],
                              "common_target_minimum_horizon": 8, "audit_window_count": 4}}
    seeds = {"master_seed": 12346, "predictor_seed": 12346, "preprocessor_seed": 1012346,
             "score_window_seed": 12347, "perturbation_seed": 12348}
    checks = {}
    for method in ("baseline", "mamba_concat", "tcn_concat"):
        a = run_case(x, graph, cfg, config, seeds, method, 20, output / method / "a", "cpu")
        b = run_case(x, graph, cfg, config, seeds, method, 20, output / method / "b", "cpu")
        differences = {
            "state_max_abs": max(float(np.max(np.abs(a[3][k] - b[3][k]))) for k in a[3]),
            "prediction_max_abs": float(np.max(np.abs(a[2] - b[2]))),
            "score_max_abs": float(np.max(np.abs(a[1].s_total_nominal - b[1].s_total_nominal))),
        }
        if method == "baseline":
            differences["partial_total_max_abs"] = float(np.max(np.abs(
                a[1].s_partial_nominal - a[1].s_total_nominal)))
            if differences["partial_total_max_abs"] > 1e-7:
                raise ValueError("Baseline partial/total parity failed")
        if max(differences.values()) > 1e-6:
            raise ValueError(f"CPU determinism failed: {method}")
        checks[method] = differences
    return {"passed": True, "formal_result": False, "scientific_evidence": False,
            "runs": 6, "iterations_per_run": 20, "data_seed": 12345,
            "checks": checks, "environment": environment("cpu")}


def protocol_plan(root: Path, protocol: str) -> tuple[Path, dict, list[dict]]:
    from phase9_lorenz_901_confirmation import load_protocol, load_run_matrix
    from phase9_audit_stageb import validate_matrix
    if protocol == "lorenz":
        path = root / "code/configs/phase9_lorenz_901_confirmation_v1.json"
        config = load_protocol(path)
        rows = load_run_matrix(config)
    else:
        path = root / "code/configs/phase9_audit_stageb_v1.json"
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        matrix = root / "code" / config["matrix_path"]
        with matrix.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = validate_matrix(rows, config=config)
    return path, config, rows


def experiment(root: Path, output: Path, args) -> dict:
    import torch
    from phase9_audit_stageb import load_dataset, model_config
    from phase9_lorenz_901_confirmation import method_seed_bundle
    from lorenz96_frozen import generate_lorenz96

    path, config, rows = protocol_plan(root, args.protocol)
    report = {"protocol": args.protocol, "config_sha256": sha256(path), "run_count": len(rows),
              "rows": rows, "executed": False, "formal_result": False}
    write_json(output / "plan.json", report)
    if not args.execute:
        return report
    if args.row is None or not 1 <= args.row <= len(rows):
        raise ValueError("Execution requires --row with a 1-based frozen matrix row")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no CPU fallback is permitted")
    row = rows[args.row - 1]
    if args.protocol == "lorenz":
        ds = config["dataset"]
        x, _, graph = generate_lorenz96(
            d=ds["d"], t=ds["T"], t_eval=ds["t_eval"], forcing=ds["forcing"],
            seed=row["data_seed"], delta_t=ds["delta_t"],
            observation_noise_sd=ds["observation_noise_sd"], burn_in=ds["burn_in"])
        seeds = method_seed_bundle(config=config, data_seed=row["data_seed"], train_seed=row["train_seed"])
        iterations = row["iterations"]
    else:
        if args.data_root is None:
            raise ValueError("Stage B execution requires --data-root; see THIRD_PARTY_DATA.md")
        x, graph, metadata = load_dataset(row["data_unit"], data_root=args.data_root, config=config)
        # Loader metadata describes the original protocol; this invocation is a reproduction.
        metadata["formal_result"] = False
        write_json(output / "dataset.json", metadata)
        seeds = {key: int(row[key]) if row[key] else None for key in
                 ("master_seed", "predictor_seed", "preprocessor_seed", "score_window_seed", "perturbation_seed")}
        iterations = config["training"]["max_iter"]
    cfg = model_config(d=x.shape[0], config=config)
    result, *_ = run_case(x, graph, cfg, config, seeds, row["method"], iterations,
                         output / "run", args.device,
                         reseed_after_initialization=args.protocol == "stageb")
    report.update({"executed": True, "row": args.row, "run": result})
    return report
