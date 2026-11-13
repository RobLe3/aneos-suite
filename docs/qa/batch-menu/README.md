# Batch menu QA, 2026-11-13

Scope: actual primary option 3 and its batch branches in options 9/10/13. This is not
four fully certified options or exhaustive menu coverage. Parent #41 retains the
remaining reachable-route denominator and unreviewed features.

## Comparator, sample and gates

Before: `dd56b829fd61d6a4d47f9be61088c1975a2aff1e`. After: the associated PR head;
artifact manifests record individual source hashes. The same six synthetic request
rows contain a candidate positive control, missing orbit, unavailable acquisition,
exact repeat, wrong identity and a long bracketed label. No real observation or
provider is substituted by that fixture.

Pre-existing #41/ADR-094 gates applied to this sample:

| Check | Before | After acceptance evidence |
|---|---|---|
| All six input occurrences visible | Missing/origin-invalid items skipped or absent | Six ordered reports, five exact labels, one explicitly reused row |
| Missing and unavailable distinct from zero | Skipped, no full request history | Inconclusive/error state with null rank and explicit error details |
| Canonical analysis | Separate fetch/detection loop | Existing bounded batch pipeline; same per-object reports and policy |
| Browsing/export | Latest-success view only | Frozen request history and explicit export scope; complete JSON/CSV |
| Readability at 80/120/160 | Baseline captured, not required to pass new gates | All rows/labels/statuses visible, width assertions, native full-precision export |
| Literal bracketed label | Legacy Rich strings could interpret markup | `Text` cells retain literal `[red]`; long value wraps |
| Scientific certainty | Origin column added no calibrated information | Uncalibrated rank, explicit unavailable inference, repeats not independent |

The first pre-fix test failed on absent batch history; the negative file-path control
passed. Two further negative tests showed that a rehashed failed report could claim
candidate/completed status. The shared report verifier now rejects those contradictions
without requiring a checksum to establish authenticity. Existing `service_unavailable`
was retained after the first full-suite run exposed its omission from the failure set.
No test is disabled or acceptance gate reduced.

## Artifacts and reproduction

Run from an installed checkout with a new output directory:

```bash
python scripts/qa_batch_menu.py --output /tmp/aneos-batch-capture-new
```

It produces before/after Rich SVG/HTML/text at all three widths, actual plain output,
the synthetic input/result and a hash manifest. The script refuses an existing
output directory. Native source code is loaded only from the fixed repository
baseline. Provider calls/persistence are replaced by explicitly synthetic controls;
network/DNS are denied while running each handler.

Captures use an **explicit xterm-256color profile**. Initial capture attempts exposed
the known TERM=dumb width clamp (#69); this is not reported as fixed or qualified.
The baseline overflow does not pass the new layout rubric; it remains in the before
artifacts. New output is required to fit each declared width. Exported SVGs remove
remote font declarations. Local raster inspection at 80/120 columns checks actual
layout, not only text tokens. Independent astrophysicist usability review is pending.

Developer-local capture directory for this review:
`/tmp/aneos-batch-qa-v3.20261113` (before final synthetic-banner addition). Final PR
comments link the regenerated artifact location and test/CI evidence. These synthetic
artifacts are not archived research measurements. Source hashes, not directory names,
identify exactly which code produced them.
