# Use the aNEOS menu

Run `python aneos.py` from the installed checkout. The primary menu uses the
shared [option catalogue](../../aneos_menu_catalog.py). The application label is
**2.2.0b2**; it is not a scientific-model or dataset version.

The menu supports Rich and plain terminal output. Navigation/banner controls are
tested at selected widths; that does not establish that every detailed view and
legacy workflow has completed behavioral or scientific QA.

## Options and prerequisites

| Choice | Displayed label | What to check before interpreting it |
|---|---|---|
| 1 | Screen one object | Requires a designation and acquisition or admissible cached evidence; rank is uncalibrated |
| 2 | Inspect screening evidence | Review inputs, proxy contributions, dependencies and unavailable inference |
| 3 | Screen a batch | Supply a designation file; examine each object's failures rather than only aggregate counts |
| 4 | Inspect orbital history | Catalogue/solution history is not proof of a manoeuvre; epochs and models matter |
| 5 | Inspect impact evidence | Scoped external Sentry results; an unlisted object does not mean zero risk |
| 6 | Inspect encounter history | Computed encounters are not observations or intentional targeting |
| 7 | Run historical polling | Check actual configured source/window and coverage; not every bulk route is restored |
| 8 | Inspect population patterns | Descriptive grouping is not calibrated significance |
| 15 | Screen orbital proximity | Pairwise catalogue prefilter, not rendezvous proof |
| 9 | Browse results | Distinguish current-session results, persisted evidence and frozen reports |
| 10 | Export results | Select an available result and preserve its provenance; avoid reusing an output path |
| 11 | Check components | Imports/reachability are not independent scientific certification |
| 12 | Start API server | For controlled local testing, prefer the explicit-loopback CLI below |
| 13 | Inspect session analytics | Session counts are not population detection accuracy |
| 14 | Read scientific help | Definitions and scientific limits, not a validated origin model |
| 0 | Exit | Return without starting another feature |

Use option 14 before interpreting screening results. **Follow-up candidate**, **not
flagged** and **inconclusive** describe the installed screening policy and evidence,
not artificial/natural ground truth. The score threshold is in ranking units.

## First object workflow

1. Set `ANEOS_DATABASE_URL` to a separate file-SQLite test database as described in
   [installation](installation.md).
2. Start the menu, choose **1**, and enter a real designation such as `1998 KY26`.
   This may query providers; source failures must remain visible.
3. Read the measurement and source-quality details. Distinguish native observations,
   fitted quantities, computed predictions and heuristic proxies.
4. Browse the result with **9** and use **10** for the available export choices.
   Preserve the evidence/policy identifiers and do not infer new observations from
   a different display of the same result.

For repeatable file-based work, use the CLI examples in the
[quick start](quick-start.md); they give an explicit designation and output path.
No example here promises a particular score or 100% measurement completeness.

## Polling differs by release

The published v2.2.0b2 quarantines the known-bad root bulk discovery paths. This
development checkout restores **bounded NASA_CAD collection** through the shared
adapters and existing ledger; it does not restore every provider's bulk interface.
See the [root CAD guide](../engineering/root-cad-workflow.md) for native row limits,
explicit object-enrichment budgets, partial coverage and replay.

A collection of predicted encounters does not, by itself, provide the orbit,
physical measurements or observations needed for a complete assessment. No browser
opens in the development direct CAD command unless explicitly requested.

## Local API and legacy menu

For API testing, use an explicit loopback address rather than relying on server
startup defaults:

```bash
python aneos.py api --dev --host 127.0.0.1 --port 8000
```

The original grouped menu remains accessible with `python aneos.py --legacy-menu`.
Its labels and option count are not promises that all historical features or
inference methods remain available. Some functions are deliberately unavailable
or controlled-deprecated to prevent fabricated measurements and certainty.

## Report a presentation or behavior defect

Include the commit/version, option and sub-option, terminal width, Rich/plain mode,
designation, source status, expected/actual result and a redacted output capture.
Check narrow and wide views as well as missing/partial/error states. A polished
normal-case screenshot is not proof that a failure state is understandable.

Full option-by-option QA remains
[issue #41](https://github.com/RobLe3/aneos-suite/issues/41), with presentation work
tracked in [#42](https://github.com/RobLe3/aneos-suite/issues/42).
