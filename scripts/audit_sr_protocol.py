"""Audit portable configs with an explicitly bound, selection-verified prior.

Run through scripts/run.py. The original auditor expects an unbound template;
only that field is normalized on a copy, after checking the actual binding.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from scripts.audit_journal_training_protocol import (
    ProtocolError, audit_dataset, data_randomness_protocol, sampler_prefix,
    sha256_file, validate_matched_configs,
)


def validate_bound_configs(candidate, baseline, selection, candidate_variant="c2"):
    if selection.get("mode") != "calibration":
        raise ProtocolError("Stage 3 requires a calibration selection record.")
    bound = candidate.get("network_g", {}).get("mae_checkpoint")
    selected = selection.get("selected_checkpoint")
    expected_hash = selection.get("selected_checkpoint_sha256")
    if not bound or not selected or not expected_hash:
        raise ProtocolError("A bound prior, selected checkpoint and SHA-256 are required.")
    bound_path = Path(bound).expanduser().resolve()
    if bound_path != Path(selected).expanduser().resolve():
        raise ProtocolError("Config prior differs from the selected checkpoint path.")
    actual_hash = sha256_file(bound_path)
    if actual_hash != expected_hash:
        raise ProtocolError("Prior SHA-256 differs from the selection record.")
    template = deepcopy(candidate)
    template["network_g"]["mae_checkpoint"] = None
    comparison = validate_matched_configs(template, baseline, candidate_variant)
    return comparison, {
        "path": str(bound_path), "sha256": actual_hash,
        "selected_iteration": selection.get("selected_iteration"),
        "normalization": "mae_checkpoint=null only in the in-memory template check",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--candidate-variant", choices=("c2", "p0"), default="c2")
    parser.add_argument("--expected-gt-images", type=int)
    parser.add_argument("--expected-source-groups", type=int)
    parser.add_argument("--expected-level-order", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    with args.candidate.open(encoding="utf-8") as stream:
        candidate = yaml.safe_load(stream)
    with args.baseline.open(encoding="utf-8") as stream:
        baseline = yaml.safe_load(stream)
    with args.prior_selection.open(encoding="utf-8") as stream:
        selection = json.load(stream)
    comparison, binding = validate_bound_configs(
        candidate, baseline, selection, args.candidate_variant,
    )
    train = candidate["datasets"]["train"]
    dataset = audit_dataset(
        train, args.expected_gt_images, args.expected_source_groups,
        args.expected_level_order,
    )
    sampler = sampler_prefix(
        dataset["logical_samples"], len(dataset["level_order"]),
        int(candidate["train"]["total_iter"]), int(train["batch_size_per_gpu"]),
    )
    sampler["level_counts_by_name"] = dict(zip(dataset["level_order"], sampler["level_counts"]))
    result = {
        "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-run protocol and prior-binding audit, not an evaluation result.",
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in (("candidate", args.candidate), ("baseline", args.baseline),
                                ("prior_selection", args.prior_selection))
        },
        "comparison": comparison, "prior_binding": binding, "dataset": dataset,
        "sampler_prefix": sampler,
        "data_randomness": data_randomness_protocol(
            int(train["num_worker_per_gpu"]), int(candidate["manual_seed"]),
        ),
        "next_check": "Run verify_journal_initialization.py to verify checkpoint metadata and tensors.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
