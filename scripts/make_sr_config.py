"""Create a portable SR config without changing the matched training protocol."""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_config(args):
    backbone = args.backbone
    if backbone == "swinir" and args.variant == "random":
        raise ValueError("The study has no SwinIR random-prior control.")
    names = {
        "drct": {
            "baseline": "train_DRCT_x4_baseline_continue_100k.yml",
            "prior": "train_RN_PD_MAE_DRCT_x4_ungated_100k.yml",
            "test": "test_RN_PD_MAE_DRCT_x4_realsr_only.yml",
        },
        "swinir": {
            "baseline": "train_SwinIR_x4_baseline_continue_100k.yml",
            "prior": "train_RN_PD_MAE_SwinIR_x4_ungated_100k.yml",
            "test": "test_RN_PD_MAE_SwinIR_x4_svsr_ungated.yml",
        },
    }[backbone]
    learned = args.variant in {"corrprior", "gated"}
    if learned and not args.prior_checkpoint:
        raise ValueError("A learned-prior variant requires --prior-checkpoint.")
    if args.mode == "train" and not args.init_checkpoint:
        raise ValueError("SR training requires --init-checkpoint for the shared backbone.")
    if args.mode == "test" and not args.sr_checkpoint:
        raise ValueError("Testing requires --sr-checkpoint (the final SR EMA checkpoint).")

    template_name = names["test"] if args.mode == "test" else names[
        "baseline" if args.variant == "baseline" else "prior"
    ]
    template = ROOT / "DRCT/options" / args.mode / "journal" / template_name
    with template.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["name"] = args.name or f"{backbone}_{args.variant}_x4_{args.mode}"
    config["output_root"] = str(Path(args.run_root).expanduser().resolve())
    config["num_gpu"] = args.num_gpu
    network = config["network_g"]
    network["use_sft"] = args.variant != "baseline"
    network["use_restoration_need_gate"] = args.variant == "gated"
    network["restoration_need_gate_mode"] = "spatial"
    network["allow_random_prior"] = args.variant == "random"
    network["zero_init_sft"] = True
    network["mae_checkpoint"] = (
        str(Path(args.prior_checkpoint).expanduser().resolve()) if learned else None
    )
    if args.variant == "baseline":
        # Keep the baseline free of candidate-only flags, as required by the
        # retained protocol auditor. DRCT/SwinIR supply their own defaults.
        for key in ("use_restoration_need_gate", "restoration_need_gate_mode",
                    "allow_random_prior", "zero_init_sft", "mae_input_upscale",
                    "mae_tile_batch_size", "mae_tile_overlap"):
            network.pop(key, None)
    config["path"]["pretrain_network_g"] = str(Path(
        args.init_checkpoint if args.mode == "train" else args.sr_checkpoint
    ).expanduser().resolve())
    config["path"]["strict_load_g"] = args.mode == "test" or args.variant == "baseline"
    data = Path(args.data_root).expanduser().resolve()
    if args.mode == "train":
        train = config["datasets"]["train"]
        train["dataroot_gt"] = str(data / "train/HR_sub")
        train["dataroot_lq"] = None
        train["dataroot_lq_levels"] = [
            str(data / f"train/LR_{level}ugly_sub") for level in (90, 80, 70, 60)
        ]
        config["mae_weight"] = 0.0
    else:
        config["datasets"] = {}
        datasets = (
            [(f"RealSR_{camera}", f"RealSR(V3)/{camera}/LR", f"RealSR(V3)/{camera}/HR")
             for camera in ("Canon", "Nikon")]
            if args.dataset == "realsr" else
            [("SVSR", "SVSR/extracted/SVSR/LR", "SVSR/GT_X4_by_LR_stem")]
        )
        for index, (name, lq, gt) in enumerate(datasets, 1):
            config["datasets"][f"test_{index}"] = {
                "name": name, "type": "PairedImageDataset",
                "dataroot_gt": str(data / "benchmarks" / gt),
                "dataroot_lq": str(data / "benchmarks" / lq),
                "io_backend": {"type": "disk"},
            }
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--variant", choices=("baseline", "random", "corrprior", "gated"), required=True)
    parser.add_argument("--backbone", choices=("drct", "swinir"), default="drct")
    parser.add_argument("--dataset", choices=("realsr", "svsr"), default="realsr")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--prior-checkpoint")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--sr-checkpoint")
    parser.add_argument("--name")
    parser.add_argument("--num-gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {args.output}")
    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(args.output)


if __name__ == "__main__":
    main()
