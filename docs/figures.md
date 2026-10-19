# Figures and result provenance

The figures are documentation assets, added at the authors' request on
19 October 2026. The manuscript and advisor documents remain in the authoring
repository. No SR output was synthesized, retouched or rerun for this update.

## Architecture

![Approved CorrPrior architecture.](assets/architecture.png)

The image is the approved `corrprior_unified_preview.png` from the authoring
package `results/pd_mae_drct/journal_drawio_final_20261014`. The corresponding
[draw.io file](assets/architecture.drawio) is included for editing. Both files
are copied byte for byte, including the encoder-weight transfer connection
between Stages 1/2 and Stage 3.

Stages 1 and 2 update the same ViT encoder, reconstruction decoder and correction
head. Stage 3 freezes the selected encoder, discards the decoder, omits the
head in ungated SR, and trains the complete DRCT plus three residual SFT blocks.
The thumbnails illustrate inputs; they are not measured SR outputs. The DRCT
backbone and MAE-style organization retain the attributions in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

To revise the diagram, open `architecture.drawio` in diagrams.net or the desktop
app, edit it, and export a PNG on a white background to `architecture.png`.
Update the asset hashes in `assets/manifest.json` after checking both versions.

## Quantitative plot

![Paired PSNR effect estimates.](assets/psnr_effects.svg)

[psnr_effects.csv](assets/psnr_effects.csv) contains the five reported effects
and paired 95% confidence intervals. The values match the authoring snapshot's
controlled comparisons and evaluation appendix, identified in
[provenance.md](provenance.md). The plot is newly rendered from those existing
numbers; it does not estimate statistics or substitute for per-image reports.

The left panel compares trained models: two camera-balanced RealSR comparisons
and one SVSR comparison with scene-clustered uncertainty. The right panel
compares the same trained CorrPrior checkpoint with its features zeroed or
spatially shifted at inference. Positive differences favor the first model or
the original aligned features, respectively. The two panels use different axis
ranges. An interval crossing zero is inconclusive, not evidence of equivalence.

Regenerate or verify the vector plot without additional dependencies:

```bash
python scripts/render_result_figure.py
python scripts/render_result_figure.py --check
```

If values change, document their experimental source, regenerate the plot and
update its CSV/SVG hashes in `assets/manifest.json`. Do not edit plotted numbers
independently of the CSV.

## Qualitative comparison

![Selected RealSR results from both cameras.](assets/qualitative_comparison.jpg)

The panel is copied unchanged from `qualitative_comparison.jpg` in the same
authoring package. Columns are the enlarged input, BSRGAN, SwinIR-L, DRCT-Base,
DRCT-RandPrior, CorrPrior and ground truth. Rows are Canon_043, Canon_011,
Nikon_033 and Nikon_039. The example-selection rule and interpretation are
documented in [results.md](results.md#qualitative-comparison).

The embedded images remain subject to the original datasets' terms. Including
these selected crops in a private documentation panel does not supply a dataset
or grant redistribution rights. Confirm figure/image reuse before public release.

## Presentation review

| Check | Evidence and decision |
| --- | --- |
| Contribution | Learned-versus-random RealSR effect is shown with its CI; no claim that SFT itself is new. |
| Clarity | Train/freeze roles, weight transfer, datasets, metric units and interval meaning are explicit. |
| Experimental strength | Gains and inconclusive comparisons are shown together; no multi-seed claim. |
| Evaluation scope | External methods have different training histories; selected crops do not establish a full-set ranking. |
| Method soundness | Fixed-checkpoint interventions are separated from trained controls; head-free SR agrees with the retained implementation. |

These checks review the presentation of existing evidence. They do not resolve
the missing experiments listed in [results.md](results.md#cost-and-limitations).

## Asset integrity

[manifest.json](assets/manifest.json) records SHA-256 values, source paths and
image dimensions. `scripts/check_release.py` verifies the hashes and admits
only listed PNG/JPEG documentation assets; arbitrary binaries, datasets,
checkpoints and manuscript bundles remain excluded.
