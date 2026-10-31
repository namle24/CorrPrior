# Figures and sources

The architecture diagram, result plot and selected RealSR comparisons are
included as documentation assets. Their sources, dimensions and SHA-256 hashes
are recorded in [manifest.json](assets/manifest.json).

## Architecture

![CorrPrior architecture.](assets/architecture.png)

[Editable draw.io source](assets/architecture.drawio) ·
[README overview](assets/overview.png)

The diagram uses the larger-font Figure 1 prepared on 29 October 2026 in
`results/pd_mae_drct/figure1_pdf_matched_20261029` of the authoring repository.
The full PNG and draw.io source are copied unchanged. The README overview
renders panels (a) and (b) from the same draw.io model; the full diagram also
shows the DRCT residual dense group and residual SFT block.

The dashed green path transfers encoder weights between prior learning and SR.
The input thumbnails are illustrative. Training and inference roles are
described in [method.md](method.md).

To edit the diagram, open `architecture.drawio` in diagrams.net, then export
a PNG with a white background. Update the overview and manifest hashes after
checking that the labels and connections remain legible.

## Quantitative plot

![Paired PSNR effect estimates.](assets/psnr_effects.svg)

[psnr_effects.csv](assets/psnr_effects.csv) contains five reported effects and
paired 95% confidence intervals from the experiment snapshot recorded in
[provenance.md](provenance.md). The plot renders those values directly.

The left panel compares trained models; the right panel intervenes on the
features of one fixed CorrPrior checkpoint. RealSR intervals account for
cameras, and SVSR intervals account for scene groups. The panels use different
axis ranges. See [results.md](results.md#controlled-comparisons) for interpretation.

Regenerate or verify the plot without additional dependencies:

```bash
python scripts/render_result_figure.py
python scripts/render_result_figure.py --check
```

When updating values, record their experimental source, regenerate the plot
and update both CSV/SVG hashes in the asset manifest.

## Qualitative comparison

![Selected RealSR results from both cameras.](assets/qualitative_comparison.jpg)

The panel is copied unchanged from `qualitative_comparison.jpg` in the
`journal_drawio_final_20261014` authoring package. Columns show the enlarged
input, BSRGAN, SwinIR-L, DRCT-Base, DRCT-RandPrior, CorrPrior and ground truth.
Rows are Canon_043, Canon_011, Nikon_033 and Nikon_039.

The selection rule and interpretation are documented in
[results.md](results.md#qualitative-comparison). The comparison crops have not
been retouched or regenerated, and remain subject to the original datasets'
terms.

## Asset integrity

`python scripts/check_release.py` verifies the manifest hashes and allows only
listed documentation assets. Dataset collections, checkpoints and manuscript
bundles are excluded from this repository.
