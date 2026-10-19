import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_sr_config", ROOT / "scripts/make_sr_config.py")
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def options(variant="corrprior", mode="train", extra=()):
    return CONFIG.parse_args([
        "--mode", mode, "--variant", variant,
        "--init-checkpoint", "checkpoints/initial.pth",
        "--prior-checkpoint", "checkpoints/prior.pth",
        "--sr-checkpoint", "checkpoints/final.pth",
        "--output", "runs/configs/test.yml", *extra,
    ])


@pytest.mark.parametrize("variant", ["baseline", "random", "corrprior", "gated"])
def test_preserves_matched_sr_training(variant):
    baseline = CONFIG.build_config(options("baseline"))
    config = CONFIG.build_config(options(variant))
    assert config["train"] == baseline["train"]
    assert config["datasets"] == baseline["datasets"]
    assert config["manual_seed"] == 0
    assert config["mae_weight"] == 0.0
    assert config["train"]["total_iter"] == 100000
    assert config["path"]["param_key_g"] == "params_ema"
    assert config["network_g"]["use_sft"] == (variant != "baseline")
    assert config["network_g"].get("allow_random_prior", False) == (variant == "random")
    assert config["network_g"].get("use_restoration_need_gate", False) == (variant == "gated")
    if variant == "random":
        assert config["network_g"]["mae_checkpoint"] is None
    if variant in ("corrprior", "gated"):
        assert config["network_g"]["mae_tile_overlap"] == 64
        assert config["network_g"]["mae_input_upscale"] == 4


@pytest.mark.parametrize("dataset,count", [("realsr", 2), ("svsr", 1)])
def test_test_configs_strictly_load_final_ema(dataset, count):
    config = CONFIG.build_config(options(mode="test", extra=("--dataset", dataset)))
    assert len(config["datasets"]) == count
    assert config["path"]["strict_load_g"] is True
    assert config["path"]["param_key_g"] == "params_ema"
    assert config["tile"] == {"tile_size": 128, "tile_pad": 16}


def test_learned_prior_is_required():
    args = options()
    args.prior_checkpoint = None
    with pytest.raises(ValueError, match="prior-checkpoint"):
        CONFIG.build_config(args)


def test_no_unreported_swinir_random_control():
    with pytest.raises(ValueError, match="no SwinIR random"):
        CONFIG.build_config(options("random", extra=("--backbone", "swinir")))


@pytest.mark.parametrize("variant,label", [("corrprior", "c2"), ("gated", "p0")])
def test_generated_pair_passes_original_protocol_checker(tmp_path, variant, label):
    candidate, baseline, selection = bound_pair(tmp_path, variant)
    result, binding = AUDIT.validate_bound_configs(candidate, baseline, selection, label)
    assert result["candidate_variant"] == label
    assert binding["sha256"] == selection["selected_checkpoint_sha256"]
    assert candidate["network_g"]["mae_checkpoint"] == selection["selected_checkpoint"]


AUDIT_SPEC = importlib.util.spec_from_file_location("audit_sr_protocol", ROOT / "scripts/audit_sr_protocol.py")
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def bound_pair(tmp_path, variant="corrprior"):
    import hashlib
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"synthetic checksum fixture, not a model")
    candidate = CONFIG.build_config(options(variant, extra=("--prior-checkpoint", str(checkpoint))))
    return candidate, CONFIG.build_config(options("baseline")), {
        "mode": "calibration", "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_iteration": 75000,
    }


@pytest.mark.parametrize("tamper", ["hash", "path", "mode", "lr", "missing_hash"])
def test_portable_audit_rejects_mismatched_binding_or_protocol(tmp_path, tamper):
    candidate, baseline, selection = bound_pair(tmp_path)
    if tamper == "hash":
        selection["selected_checkpoint_sha256"] = "0" * 64
    elif tamper == "path":
        candidate["network_g"]["mae_checkpoint"] = str(tmp_path / "different.pth")
    elif tamper == "mode":
        selection["mode"] = "partial"
    elif tamper == "lr":
        candidate["train"]["optim_g"]["lr"] *= 2
    else:
        selection.pop("selected_checkpoint_sha256")
    with pytest.raises(AUDIT.ProtocolError):
        AUDIT.validate_bound_configs(candidate, baseline, selection)


def test_portable_audit_cli_on_synthetic_inventory(tmp_path):
    import json
    import subprocess
    import sys
    import yaml

    candidate, baseline, selection = bound_pair(tmp_path)
    roots = [tmp_path / name for name in ("HR_sub", "LR_90ugly_sub", "LR_80ugly_sub",
                                         "LR_70ugly_sub", "LR_60ugly_sub")]
    for root in roots:
        root.mkdir()
        # The protocol auditor inventories filenames; it does not decode images.
        (root / "source_s001.png").touch()
    for config in (candidate, baseline):
        config["datasets"]["train"]["dataroot_gt"] = str(roots[0])
        config["datasets"]["train"]["dataroot_lq_levels"] = [str(p) for p in roots[1:]]
        config["train"]["total_iter"] = 2
    candidate_path, baseline_path = tmp_path / "candidate.yml", tmp_path / "baseline.yml"
    candidate_path.write_text(yaml.safe_dump(candidate))
    baseline_path.write_text(yaml.safe_dump(baseline))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    output = tmp_path / "audit.json"
    subprocess.run([
        sys.executable, "scripts/run.py", "scripts/audit_sr_protocol.py",
        "--candidate", str(candidate_path), "--baseline", str(baseline_path),
        "--prior-selection", str(selection_path), "--expected-gt-images", "1",
        "--expected-source-groups", "1", "--output", str(output),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["dataset"]["logical_samples"] == 4
    assert report["sampler_prefix"]["consumed_samples"] == 8
    assert report["prior_binding"]["sha256"] == selection["selected_checkpoint_sha256"]
