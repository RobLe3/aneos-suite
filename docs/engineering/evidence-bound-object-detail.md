# Evidence-bound object details

## A. Engineering changes

V2 catalogue details previously called orbital-field presence “data completeness”;
this could imply a complete physical dataset. Rounded sizes/periods and sliced dates
also concealed useful precision. Screening details trusted mutable flat result fields
and did not explain the reference population or hand-set inclination enhancement.

`aneos_core/analysis/object_review.py` now freezes the typed NEO snapshot, canonical
assessment and policy. Whole-review replay rejects edited conclusions, traces, counts
and mismatched objects even after a new self-hash. The existing screening calculation
records its actual terms; its formula/order/threshold and v1 assessment payload are
unchanged. Interpretation distinguishes follow-up ranking, not-flagged and abstention.
A non-flagged object is not thereby natural or “perfectly normal”.

Rich and plain views show a fixed 15-field catalogue-presence profile, native values,
small-body display conversions, literal object names, unsliced timestamps and explicit
missing data. Three screening inputs are a different denominator. Overall completeness
and source coverage are not established; no unavailable field is fabricated. A1/A2/A3
and covariance/display-epoch cautions remain visible. Explanations identify the fixed
prototype population, size/photometry dependencies and manual orbital enhancement.
Graphics are labeled squared-ranking contributions, not probabilities.

V2 detection builds the review using the actual source snapshot. Detail playback
requires canonical identity/result binding (case-only browser keys are supported,
not arbitrary alias resolution). Assessment-only legacy data explicitly lacks the full
snapshot. JSON/CSV exports preserve unrounded review data and use exclusive creation.
Flat scientific claims are unavailable; legacy export records are marked unverified.
Legacy acceleration provenance remains retained rather than discarded.

## B. Scientific assessment and limits

This is computational provenance and presentation, not independent source verification,
scientific calibration, a source-enrichment ledger or a physical-origin investigation.
The legacy 800 m prototype diameter centre is explained, not silently replaced. No
ranking is tuned for KY26. Historical November 12 results/data are unchanged. The
illustrative small-body fixture resembles the rounded supplied inputs but is explicitly
synthetic, not a new observation or reproduction of the missing original full precision.
No Phobos 1 or natural-force model has been fitted by this change.

## C. Architecture and files

ADR-080/DDD place the review in existing analysis/presentation responsibilities. No new
bounded context, service or UI is introduced. Changed integration: `aneos_menu_v2.py`;
new module, renderer tests, QA generator and artifacts are linked here. Existing test
seams now capture real output instead of intercepting a retired panel helper; identity
is taken from the actual canonical payload, not a mismatched test label.

## D. Remaining work

#40: normalized multi-source local ledger, pollability/source denominator, identity,
ambiguity, atomic uncertainties and enrichment. #41: all remaining menu prerequisites,
full positive/negative runtime paths and legacy/pipeline detail views. #42: broader
report/API integration, historical playback and genuine physical insights/plots.
#43: fitted hypotheses, conditional populations and independent calibration. Existing
asynchronous generic menu persistence is not made durable by a successful export;
#15 remains open. No whole-application readiness or five-sigma claim is made.

## E. Reproduction and QA

Use a disposable database, not an existing research store:

```sh
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-review-check.db python -m pytest tests/test_object_evidence_review.py -q
python scripts/qa_object_review.py --output /tmp/aneos-object-detail-captures
```

The capture directory must not exist. Committed captures:
[object-detail/2.2.0.dev0](../qa/object-detail/2.2.0.dev0/). Three synthetic controls
(small-body illustration, follow-up, missing data) have actual Rich text/HTML/SVG at
80/120/160 columns, full JSON and SHA-256 inventory/source hashes. These are terminal
renders, not provider or scientific validation. Remote-font rules are removed using
the existing QA helper. Raster inspection uses local rsvg-convert; no new application
dependency. JSON retains original precision; readable tables use nine significant
figures and show native units. Full-suite/CI results are recorded in the PR, with
failures and migrations disclosed, not treated as absent defects.

### Verification record

The initial targeted run exposed five obsolete/mismatched test seams, then actual
JSON/CSV round-trip tests caught envelope/comment handling in the tests. The first
full run also caught the hidden acceleration-inventory regression; all axes and
covariance cautions were restored rather than removing that assertion. A deep-dotted
coverage target triggered premature scientific-package imports locally (NumPy reload /
SciPy import failure); top-level source discovery with a report include, consistent
with existing component gates, avoids that startup path. No scientific dependency,
assertion or quality threshold was disabled. Final component run:29 passed,95.34%
statement coverage against the new85% gate. All tests use disposable databases and
synthetic inputs. Rich-generated SVG whitespace is preserved in the hashed artifacts;
source diff checks exclude generated exports only.
