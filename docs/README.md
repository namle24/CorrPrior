# Documentation

| Guide | Contents |
| --- | --- |
| [Method](method.md) | Prior learning, correction supervision and transfer to DRCT |
| [Data](data.md) | Dataset layout, degradation generation and source-disjoint splits |
| [Training](reproduction.md) | Commands for all three stages and matched controls |
| [Evaluation](evaluation.md) | Inference, metrics and paired statistical comparisons |
| [Results](results.md) | Benchmark tables, ablations, diagnostics and qualitative examples |
| [Reproducibility](reproducibility.md) | Artifact availability, environments and validation commands |
| [Figures](figures.md) | Editable architecture, figure sources and asset checks |
| [Provenance](provenance.md) | Source history and experiment snapshot |

## Code layout

```text
PD_MAE_SR/          Prior architecture, data loaders and prior training
DRCT/drct/         SR architectures, SFT integration and SR training
DRCT/options/      Experiment configuration templates
DRCT/scripts/      Metrics, paired statistics and protocol checks
tools/degradation/ Degradation pipeline
scripts/           Launcher, config generator and repository checks
tests/             Configuration and integration tests
docs/              Guides, results and figure assets
```

Historical names such as `PD_MAE_SR` and `RN_PD_MAE` are retained for import
and checkpoint compatibility. The method is named CorrPrior in the paper.
