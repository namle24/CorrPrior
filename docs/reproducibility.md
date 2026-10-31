# Reproducibility

## Available artifacts

The repository includes the prior and SR implementations, training and evaluation
entrypoints, configuration templates, statistical tools and automated tests.
Reported aggregate results and figure sources are in [results.md](results.md)
and [figures.md](figures.md).

Numerical reproduction also requires artifacts outside this repository:

| Artifact | Availability |
| --- | --- |
| Training and benchmark datasets | Obtain from the original providers; see [data.md](data.md) |
| Prior and SR checkpoints | Public downloads are not yet available |
| Pretrained backbone and metric weights | Obtain separately for training or evaluation |
| Exact crop/source inventories and per-image experiment records | Retained by the authors |

Selected images embedded in documentation figures are included. Dataset
collections, full-resolution SR outputs and manuscript files are kept separately.

## Environment

The portable test setup uses Python 3.12, PyTorch 2.5.1, torchvision 0.20.1,
BasicSR 1.4.2 and timm 0.9.16. These versions are pinned in
[`requirements.txt`](../requirements.txt). For the complete Linux CPU test
environment, install [`requirements-cpu-validation.txt`](../requirements-cpu-validation.txt)
in a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements-cpu-validation.txt
```

For CUDA, install the matching PyTorch and torchvision packages for the target
GPU and driver before the remaining dependencies. The
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v251)
lists the 2.5.1/0.20.1 builds. Newer GPUs may require a newer build.

The original training audit records `2.13.0.dev20260513+cu132`. The portable
dependency pins are a validation environment, not a reconstruction of that
training environment. CPU tests cover implementation behavior; GPU training,
runtime and numerical reproduction require validation on the target hardware.
Apple Silicon GPU training has not been validated.

[`scripts/run.py`](../scripts/run.py) sets the repository import paths and
provides the legacy torchvision import alias required by BasicSR. The alias
comes from [`run_legacy_basicsr_entrypoint.py`](../DRCT/scripts/run_legacy_basicsr_entrypoint.py)
and does not change weights, losses or image transformations.

## Validation

After installing the CPU validation environment, run:

```bash
python -m pytest -q
python scripts/check_release.py
python scripts/render_result_figure.py --check
```

The tests cover data loading, checkpoint selection, spatial alignment, EMA
synchronization, protocol checks, paired statistics and config generation.
The repository checker verifies retained source hashes, Python syntax,
documentation links and the documented asset inventory. The plot check ensures
that the SVG matches the reported CSV values.

[GitHub Actions](https://github.com/namle24/CorrPrior/actions/workflows/tests.yml)
runs these checks on pushes and pull requests. The initial clean-clone
validation record is in [validation.md](validation.md).
