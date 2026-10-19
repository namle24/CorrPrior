# Reported results and their scope

These values summarize the authors' existing experiments, not runs performed
during repository extraction. The manuscript is kept separately by request.
The snapshot identifier is recorded in `provenance.md`.

## Controlled comparisons

| Comparison | Dataset / endpoint | Candidate minus control | Paired 95% CI | Interpretation |
| --- | --- | ---: | --- | --- |
| CorrPrior vs DRCT-RandPrior | Camera-balanced RealSR PSNR | +0.1321 dB | [0.0903, 0.1827] | Supports learned information beyond an equally sized random branch |
| CorrPrior-Gated vs CorrPrior | Camera-balanced RealSR PSNR | -0.0017 dB | [-0.0065, 0.0031] | No supported additional gain from score gating; not an equivalence result |
| CorrPrior vs DRCT-Base | Scene-clustered SVSR PSNR | +0.0030 dB | [-0.0122, 0.0184] | Inconclusive PSNR gain |
| CorrPrior vs DRCT-Base | Scene-clustered SVSR DISTS | -0.002322 | [-0.002834, -0.001797] | Secondary metric benefit, not a replacement for the PSNR endpoint |

On RealSR, absolute PSNR is 27.4307 dB for CorrPrior, 27.2985 for
DRCT-RandPrior, 27.2828 for DRCT-Base and 27.4290 for CorrPrior-Gated.
Effect sizes use unrounded per-image measurements, so subtracting rounded
table cells can give a slightly different final digit.

## Mechanism diagnostics

The selected Stage 2 head predicts a mean correction of 0.0038 on clean inputs.
Within-image correlations with correction targets range from 0.6940 to 0.7813
across degraded inputs. On 300 unseen spatial mixtures, aligned predictions
correlate more strongly with targets than spatially shifted predictions.

Post-hoc interventions on the fixed CorrPrior SR checkpoint reduce RealSR PSNR
by 0.3194 dB when features are zeroed and 0.3678 dB when features are shifted.
These are intervention effects on one trained model, not gains over retrained
baselines or independent confirmation of the primary result.

## Cost and limitations

The DRCT integration adds 415,800 trainable SFT parameters and 22,178,232 total
parameters including the frozen prior. Freezing the encoder does not remove
its inference cost. Perceptual metrics do not uniformly favor CorrPrior, and
no human preference study was conducted.

The principal DRCT comparisons use one seed at 4x scale. The study does not
isolate the effect of Stage 2, each prior-learning loss or the fixed correction
mapping parameter. Exploratory SwinIR transfer is mixed and has no matched
random-prior control. Do not infer backbone-independent improvement.

## Historical experiment identifiers

| Identifier | Paper name | Role |
| --- | --- | --- |
| B0 | DRCT-Base | No prior branch |
| C0 | DRCT-RandPrior | Frozen random encoder, matched branch capacity |
| C2 | CorrPrior | Main ungated learned-prior model |
| P0 | CorrPrior-Gated | Direct correction-score gating control |

The initial gated study did not meet its DISTS criterion. The fidelity follow-up
and later ungated analysis must not be presented as if all endpoints and model
choices were part of a single prospective experiment. Selection rules and
exploratory limits are described in the manuscript. Historical analysis utilities
are retained for traceability; their presence does not mean all conditional
experiments were executed.
