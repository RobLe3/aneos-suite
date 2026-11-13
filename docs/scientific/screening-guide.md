# Reading an aNEOS result

This development build supports reproducible research screening. It is not an
operational warning service, an independently calibrated discovery procedure or
an origin classifier. A successful query is not proof of complete observations.

## Start with the evidence

Check the designation and catalogue identity, source and retrieval time, observation
arc, orbit solution and epoch. Distinguish observations from fitted parameters and
computed encounters. Preserve native units, time scales, uncertainty and covariance
labels. A catalogue query date or predicted encounter is not an observation date.

Missing values mean not available, never zero. Conflicting identities or measurements
need resolution or an explicit alternative; more rows are not necessarily more
independent evidence. Completeness is meaningful only against a declared required
field set. Source coverage, enrichment completion and analysis readiness are different.
The target is all obtainable required data, with unresolved gaps shown honestly.

## What the screening result means

| Output | Meaning | What it does not establish |
| --- | --- | --- |
| Screening score | Uncalibrated ranking under the stated engine/reference | Gaussian significance or origin probability |
| FOLLOW_UP_CANDIDATE | Ranking meets the recorded policy threshold | An artificial object or a validated physical anomaly |
| NOT_FLAGGED | This policy did not flag the supplied evidence | Perfectly normal, natural origin or no need for other research |
| INCONCLUSIVE | Evidence or model prerequisites do not support the operation | A negative scientific finding |
| Null significance/probability | No justified calibrated result is available | Zero risk, zero anomaly or zero probability |

The existing rank includes dependent proxies and prototype population assumptions.
Size, albedo and absolute magnitude must not be treated as independent origin evidence.
The report records evidence, reference/policy identifiers and limitations; compare
like-for-like inputs and versions rather than interpreting a rank as a probability.

## Physical dynamics and comparative explanations

A1, A2 and A3 are radial, transverse and out-of-plane force-model coefficients.
Check their source units, uncertainties, estimated/fixed/considered status and orbit
solution. A coefficient or improved fit does not identify a cause. Assess compatible
natural, human-technology and measurement/model-artifact explanations against the
same observations, with uncertainty, parameter complexity and predictive checks.
A human spacecraft hypothesis is not extraterrestrial evidence.

No valid physical-origin fit comparison or calibrated origin probability is installed
in the current normal screening path. Keep unsupported conclusions unresolved.
A five-sigma claim needs a validated null and the full search/error-control procedure;
no fixed ranking threshold supplies that evidence.

## Population and impact views

Population neighbourhoods and native-TDB computed encounter gaps are descriptive.
They do not establish a common origin, deliberate coordination, measured acceleration
periodicity or calibrated network significance. Cross-object temporal correlation
requires compatible paired observations, not one fitted scalar per object.

JPL Sentry evidence concerns the source model's tabulated Earth events and scope.
Not listed is not zero impact probability. Impact evidence and origin screening
are separate; a screening rank cannot supply a collision probability or risk class.

## Reproduce, then investigate

Save the JSON/Markdown report, exact inputs/configuration, evidence/policy/report IDs,
source solution/epoch and software identity. Compare archived assessments rather
than overwrite them with later measurements. A completed execution means work was
stored, not that every scientific prerequisite passed. Check missing evidence and
follow-up needs before deriving a physical conclusion.

Detailed current contracts: docs/scientific/VALIDATION_INTEGRITY.md and the scoped
reports in docs/engineering/. The older scientific-documentation.md is historical
prototype material, not current validated methodology. Full object explanations,
menu QA, source enrichment and independent evaluation remain active work (#40–#46).
