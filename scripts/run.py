"""Run a retained research entrypoint from the repository root.

The compatibility alias restores a removed torchvision import used by BasicSR.
It does not change the architecture, checkpoint keys, losses, or input tensors.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", help="Python file relative to the repo root")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entrypoint = (ROOT / args.entrypoint).resolve()
    if ROOT not in entrypoint.parents or entrypoint.suffix != ".py":
        parser.error("entrypoint must be a Python file inside this repository")
    if not entrypoint.is_file():
        parser.error(f"entrypoint does not exist: {entrypoint}")
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "DRCT"), str(ROOT)]
    from scripts.run_legacy_basicsr_entrypoint import install_compatibility_modules
    install_compatibility_modules(ROOT)
    sys.argv = [str(entrypoint), *args.arguments]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
