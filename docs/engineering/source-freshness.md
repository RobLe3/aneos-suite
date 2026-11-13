# Source receipt review and explicit refresh

**Version:** 2.2.0.dev0, unreleased draft stack. **Issues:** #66, #40.

A recently downloaded catalogue can still contain old observations. aNEOS now keeps
those questions separate: receipt review asks when an adapter was read, not whether
an orbit fit is current enough for a scientific hypothesis test.

## Using the menu

The existing legacy multi-source polling/database-enrichment workflow reviews stored
source receipts after a successful enrichment. It displays the object, source, last
outcome, receipt-age status, original adapter-read label and retry decision. It does
not probe provider health merely to display this table.

The review covers the fixed installed object-adapter profile: SBDB, NEODyS, MPC,
Horizons and bounded CAD. Known aliases use the same canonical names. Configured
unsupported sources remain visible; they are not invented compatibility columns or
silently removed from the source plan.

For each review:

- **Recent adapter read:** the last usable typed/explicit-empty response is at most
  30 days old. This says nothing about the observation epoch or source accuracy.
- **Stale adapter read:** that receipt is older than 30 days.
- **Unknown:** no usable explicit-offset timestamp is available. Old SQLite
  compatibility timestamps remain unverified; a naive time is never assumed UTC.
- **Clock ahead:** a source receipt is later than the review clock. The system does
  not call this recent evidence. Append-order clock regression is reported in JSON.
- **Failed-attempt cooldown:** a failed adapter attempt was recorded within the
  past hour. An unverified failed-attempt clock cannot bypass that cooldown check.

These intervals are operational defaults, not provider policy or scientific lifetime
estimates. Changing them requires a versioned policy change and regression tests.

The inspector reviews at most 50 distinct stored objects, sorted by exact designation,
including query-only failures with no materialized object row. A truncated review is
labelled; the count is not a whole-database total. Histories for included objects are
replayed in full, so very long histories can make this read expensive. There is no
global alias-equivalence claim.

## Requesting a refresh

When eligible objects exist, the menu offers an **explicit whole-plan adapter
refresh**, bounded to ten selected existing objects. Stored eligibility is checked
again before each object. If a planned installed source is in cooldown, that object
is deferred rather than reducing its source denominator. Other provider admission,
pacing and cooldown rules still apply at execution time. This read/check is not a
distributed scheduler lock; concurrent source admission remains the transport's job.

The request bypasses the combined query cache through the ordinary DataFetcher.
It does not promise that a provider SDK bypasses its own cache. Successful execution
therefore means a **new committed adapter query**, not a new astronomical observation
or a guaranteed new wire response. Ordinary cached-query replay retains the original
query/receipt and does not slide its adapter-read age.

The result distinguishes:

| Result | Meaning |
|---|---|
| Recorded full plan | Every planned adapter contributed typed material. |
| Recorded partial | A new query was committed, but only some adapters contributed. |
| Recorded no data | All source outcomes were committed, with no usable material. |
| Deferred | Stored policy does not currently admit this refresh, or it is no longer due. |
| Failed | A new committed query could not be verified; no success is claimed. |

Even full response-profile contributions are **not 100% scientific completeness**.
A valid empty response is a known absence, not an invented measurement. Older source
assertions and previous scientific assessments are not overwritten. Compatibility
caches may be updated by the existing repository; their row-value history is retained,
not a byte-for-byte database backup. Cache expiry/clear still evicts cache entries.

## Reproduce the bounded QA

Use a disposable database and installed dependencies; the following tests do not
request live provider data:

```bash
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-freshness-tests.db \
  python -m pytest tests/test_source_freshness.py -q \
  --cov=aneos_core --cov=aneos_api \
  --cov-config=tests/coverage-source-freshness.ini --cov-fail-under=90

ANEOS_DATABASE_URL=sqlite:////tmp/aneos-freshness-render.db \
  python scripts/qa_source_freshness.py --output /tmp/aneos-new-freshness-captures
```

The capture command refuses an existing output directory. It runs the actual receipt
menu against disposable SQL with three synthetic cases at 80/120/160 columns, and
writes JSON, text, HTML, SVG and source/artifact hashes. Captures are examples, not
real astronomical evidence or proof that every menu feature has been exercised.

## Scope still open

Automatic selective backfill after a partial cached query remains #40. Full provider
pollability/configuration/health and the broader legacy polling/deduplication workflow
remain #16/#40. In particular, inspecting/refreshing stored objects does not repair
all candidate-discovery polling or establish an exhaustive source list. Full menu QA,
observational model fitting, independent calibration and scientifically defensible
five-sigma claims remain separate work. See ADR-089 and the chronology guide for the
boundary between native source dates and local storage receipts.

During graphical QA, #69 exposed a shared capture defect under `TERM=dumb`: old
forced-terminal SVG/HTML captures could all be 80 columns despite their requested
width labels. This change corrects the helper and adds an actual SVG-geometry
regression. The new receipt-review bundle is captured with the correction. Older
artifacts are not overwritten or retroactively credited; see the [QA evidence note](../qa/README.md).

## Delivery evidence for this block

The local offline suite passed **1,639 tests**, with eight network tests excluded;
core coverage **51.32%**, unchanged 30% floor. The **70 new cases** cover this receipt-
review/refresh component at **100% statement coverage** against its new 90% gate.
Existing gates are retained. This includes positive SQL refreshes and deferred,
no-data, persistence-failure, forged/old receipt and cleanup-failure paths; coverage
is bounded engineering evidence, not proof of universal reliability.

Nine actual SQL-backed menu captures and their source/artifact hashes were checked;
80/120/160 layouts were visually inspected after correcting #69. A separate regression
checks coloured SVG geometry under `TERM=dumb` and matches HTML text to plain text.
Final published-head CI/Docker results are recorded in the PR rather than inferred
from this local result. No independent scientific calibration or complete menu audit
is claimed.

A final bounded-selection regression also distinguishes a requested object that is
present but outside the review limit from a genuinely absent designation. Reports
expose `requested_designations_not_reviewed` separately; truncation is not missing
data. This correction is covered by the **70-case** component suite. Final-head CI is
authoritative for the published correction.
