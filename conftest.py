"""Make the retained research package layout available to portable tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "DRCT"), str(ROOT)]

from scripts.run_legacy_basicsr_entrypoint import install_compatibility_modules

install_compatibility_modules(ROOT)

import torch

torch.set_num_threads(2)
