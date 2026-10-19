# Reproducibility and release status

## What is included

- The paired prior architecture, correction target/loss implementation, data
  loaders, source-disjoint selection logic and training/evaluation entrypoints.
- DRCT integration and matched baseline, random-prior and gated variants;
  exploratory SwinIR integration.
- Portable experiment templates, a config generator, paired metric/statistical
  tools, protocol checks and selected regression tests.
- Aggregate reported results and code provenance, without the manuscript.
- The approved architecture figure and editable draw.io source, a quantitative
  effect plot, and the existing selected RealSR comparison panel. Figure
  provenance and reuse boundaries are recorded in [figures.md](figures.md).

## What is not included

- Dataset collections, original crop inventories and generated degradation
  corpora. Only selected thumbnails/crops within documentation figures are included.
- Prior, backbone, SR, VGG or perceptual-evaluation weights.
- Full-resolution outputs, private per-image reports, credentials or server scripts.
- Manuscript PDFs/LaTeX, advisor documents, old paper bundles and other research projects.

Consequently, cloning the repository alone cannot reproduce the reported
numbers. The authors must supply the missing artifacts or approve their release.
Existing dataset and model licenses still apply.

## Environment

The portable test setup uses Python 3.12, PyTorch 2.5.1, torchvision 0.20.1,
BasicSR 1.4.2 and timm 0.9.16. The selected versions are recorded in
`requirements.txt`; these are not presented as the original training lockfile.
The complete installed CPU validation package set is retained in
`requirements-cpu-validation.txt`. To recreate that test environment, install
it with `python -m pip install -r requirements-cpu-validation.txt` in a fresh
Python 3.12 environment. Do not use this CPU-specific file for GPU training.
The retained protocol audit reports the original training torch version as
`2.13.0.dev20260513+cu132`. The complete original environment has not been
reconstructed in this extraction.

For CPU checks, use the README installation commands. For CUDA, choose a build
compatible with the target GPU and driver; the [official PyTorch installation
matrix](https://pytorch.org/get-started/previous-versions/#v251) lists the matching
2.5.1/0.20.1 packages. A successful CPU test does not validate CUDA kernels,
training speed, memory use or exact numerical reproduction. A newer GPU may
need a different PyTorch/CUDA build and a new validation record.

`scripts/run.py` adds a compatibility alias for the old BasicSR import of
`torchvision.transforms.functional_tensor.rgb_to_grayscale`. The source of that
shim is retained in `DRCT/scripts/run_legacy_basicsr_entrypoint.py`. No weights,
losses or image transformations are replaced. Mac/Apple Silicon GPU training
has not been validated; the repository can still be read and edited on a Mac.

## Validation

```bash
python -m pytest -q
python scripts/check_release.py
python scripts/render_result_figure.py --check
```

The release checker verifies retained source hashes, Python syntax, documentation
links, and the absence of obvious forbidden artifacts and credential patterns.
It is a bounded audit, not a guarantee that no sensitive information exists.
The test suite covers loaders, checkpoint selection, alignment, EMA auxiliary
synchronization, protocol validation, paired statistics and config generation.
See `validation.md` for the checks actually executed for this extraction.

## Public-release checklist

- [ ] All authors approve the code and timing of publication.
- [ ] Select a project-wide license after checking source provenance and all
  upstream restrictions. The MAE implementation requires particular attention.
- [ ] Decide which checkpoints may be redistributed; publish verified hashes
  and working links if approved.
- [ ] Provide or document access to the exact crop/source manifests needed
  for numerical reproduction without violating dataset terms.
- [ ] Record the original GPU environment and validate the documented training
  and inference commands on the intended hardware.
- [ ] Confirm permission to redistribute the dataset-derived thumbnails and
  comparison crops in the documentation figures.
- [ ] Update publication status, paper link and citation only when confirmed.
- [ ] Review all files and commit history before changing visibility to public.

The repository remains private until the author explicitly approves publication.
