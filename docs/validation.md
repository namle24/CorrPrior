# Extraction validation

Date: 19 October 2026. Validation uses an isolated Linux CPU environment with
Python 3.12.3, torch 2.5.1+cpu and torchvision 0.20.1+cpu. It does not modify the
original training environment.

- Final local suite: **116 tests passed**, none skipped. This includes the
  retained regression tests, portable config checks, prior tensor smoke tests,
  SFT identity/gradient checks and a synthetic end-to-end protocol audit.
- A fresh `git clone --no-local` outside the original workspace also passed all
  116 tests and the release audit using the isolated CPU environment. No files
  from the parent workspace are needed for those checks.
- Dependency consistency check: all 57 installed packages compatible. The
  exact Linux CPU package snapshot is in `requirements-cpu-validation.txt`.
- Nine command-line entrypoints opened successfully with `--help`, including
  prior training, selection, SR train/test, metrics and both protocol auditors.
- The config generator produced an ignored local evaluation config without
  loading weights or running inference.
- Release audit: 105 source files checked; all 77 retained-source hashes match.
  Python syntax and local Markdown links pass. The bounded artifact/credential
  scan flagged no manuscript, dataset, weight, oversized binary or credential
  patterns in the export.

Known non-failing warnings: BasicSR's deprecated SciPy import and PyTorch's
future change to `torch.load`. Only load trusted checkpoints; the original
checkpoint-loading behavior was retained for compatibility.
Imported source/license files retain their original whitespace and line endings;
the initial import therefore has legacy `git diff --check` whitespace warnings.

Not performed: GPU training, full benchmark inference, a new PSNR/SSIM result,
perceptual model downloads/evaluation, Mac GPU validation, or a complete
reconstruction of the original GPU software environment.
