# 1998 KY26 source register

First inspection: 2026-11-12. This is a loading register, not a completed
literature review. No unavailable measurement values have been inferred.

| ID | Primary source | Loaded scope | Remaining work |
|---|---|---|---|
| HIST-20261112 | User-supplied baseline retained in `tests/fixtures/ky26_baseline.json` | Recorded output and approximate inputs | Original full-precision provider responses, orbit ID/epoch and exact execution configuration |
| SBDB-CONTRACT | [JPL SBDB documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html) | Provider contract inspected during engineering review | Validate current KY26 response separately; preserve A1/A2/A3 sigma/kind/units and labelled covariance; never overwrite historical baseline |
| SENTRY-20261112 | [JPL Sentry object API](https://ssd-api.jpl.nasa.gov/sentry.api?des=1998%20KY26), [dated snapshot](sentry/20261112T211912Z/report.md) | One current guarded query at 2026-11-12T21:19:13.177037Z; API 2.0 returned not found | Not listed is not zero impact risk or evidence about origin; no local impact/orbit fit or universal future-time conclusion |
| DARK-2023 | [PSJ DOI 10.3847/PSJ/acb697](https://doi.org/10.3847/PSJ/acb697) | DOI access blocked by retrieval tool | Obtain primary full text/tables; extract KY26 A3 and other components, errors, units, solution and fit context |
| PHYSICAL-2025 | [Nature Communications DOI](https://doi.org/10.1038/s41467-025-63697-4), [author article in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12446490/) | Publisher retrieval failed; alternate primary full text located, not yet fully extracted | Verify preferred and alternative size/spin/pole/shape/albedo fits and uncertainties |
| PHOBOS-2026A | [arXiv:2606.01288v2](https://arxiv.org/abs/2606.01288v2) | Abstract/submission history inspected; revised 2026-06-02 | Load full trajectory reconstruction, covariance and propellant assumptions; distinguish feasible fitted transfer from identity proof |
| PHOBOS-2026B | [arXiv:2609.24283v2](https://arxiv.org/abs/2609.24283v2), [full text](https://arxiv.org/html/2609.24283v2) | Abstract/history and full-text availability inspected; revised 2026-09-23 | Extract actual observations, nuisance/free parameters, likelihood, residual covariance, pole conventions and predictive tests |
| THERMAL-2026 | [Icarus DOI 10.1016/j.icarus.2026.117361](https://doi.org/10.1016/j.icarus.2026.117361) | Publisher metadata located; full text retrieval denied | Obtain author manuscript/supplements; identify which volatiles, depths, histories and assumptions are constrained |
| JAXA-2031 | [JAXA 2026 mission topic](https://www.hayabusa2.jaxa.jp/en/topics/2026_TorifuneCamp2s_e/), [ISAS mission page](https://www.isas.jaxa.jp/en/missions/spacecraft/current/hayabusa2.html) | Official current search excerpts inspected | Preserve dated mission source before prediction freeze; schedule is a plan, not an observation |

## Interpretation boundaries

The first Phobos preprint proposes a trajectory/propulsion hypothesis. The second
reports fit improvements using spacecraft-shaped radiation-pressure models.
Neither abstract establishes identity; no aNEOS observational refit or predictive
validation has been performed. Human technological origin is distinct from an
extraterrestrial-origin hypothesis. [First preprint](https://arxiv.org/abs/2606.01288v2),
[second preprint](https://arxiv.org/abs/2609.24283v2).

The official 2026 JAXA topic describes arrival at KY26 in July 2031. Any frozen
prediction register must retain the dated schedule and be revisited without
rewriting earlier predictions. [JAXA mission topic](https://www.hayabusa2.jaxa.jp/en/topics/2026_TorifuneCamp2s_e/).

Natural, human-technological and orbit-fit-artifact explanations remain to be
compared on compatible observations. An absent visible coma does not by itself
exclude sublimation; a constraint on water ice does not exclude every natural
mechanism. Missing full-text extraction remains missing evidence, not agreement
or contradiction.

Case loading and the complete ten-part investigation are tracked in
[issue #7](https://github.com/RobLe3/aneos-suite/issues/7); observational modelling
and hypothesis comparison are [#8](https://github.com/RobLe3/aneos-suite/issues/8)
and [#9](https://github.com/RobLe3/aneos-suite/issues/9).
