# aNEOS: Advanced Near Earth Object Suite

**v2.2.0b2 | Beta preview | Engineering test release, not scientific validation**

Investigate Near-Earth Objects with source-linked measurements, reproducible
screening and readable scientific reports. aNEOS helps identify objects worth
further study; it does not determine their origin from an anomaly score.

The long-term research goal is to test technological explanations, including human
artifacts, against natural forces and measurement/model errors. The current suite
does not provide the complete observational model comparison needed to settle those
questions. An unusual object can remain unresolved or have a natural explanation.

- [Install and run](#install-and-run)
- [Understand a result](#understand-a-result)
- [Inspect evidence and history](#evidence-and-history)
- [Documentation](docs/README.md) · [Roadmap](ROADMAP.md) ·
  [Scientific reliability programme](https://github.com/RobLe3/aneos-suite/issues/2)

## Release and development status

The published **v2.2.0b2** is an engineering beta for per-object acquisition,
evidence storage, screening and reporting. Its root legacy bulk discovery remains
quarantined. Use the [beta release guide](docs/releases/2.2.0b2.md) for that tag.

This development checkout additionally contains the bounded
[root CAD workflow](docs/engineering/root-cad-workflow.md): committed encounter
collections, source-linked physical fields and explicitly budgeted enrichment.
Those changes are under review in [PR #85](https://github.com/RobLe3/aneos-suite/pull/85),
[PR #89](https://github.com/RobLe3/aneos-suite/pull/89) and
[PR #90](https://github.com/RobLe3/aneos-suite/pull/90); they are not in the beta tag.
Other root bulk discovery choices remain unavailable.

The application version does not identify the scoring policy, input dataset or
API schema. Keep the report's separate reproducibility identifiers when comparing
results from different installations.

## What you can do

| Task | Current scope |
|---|---|
| Screen a named object | Shared evidence-based screening through the CLI, menus and integrated REST routes; score is an uncalibrated ranking |
| Inspect measurements | Source outcomes, orbital and physical fields, native A1/A2/A3 and covariance when supplied; missing values stay missing |
| Preserve local evidence | Transactional file-SQLite acquisition receipts, source-specific assertions and replayable projections |
| Read and export results | Rich/plain terminal views, versioned JSON, Markdown and existing dashboard loaders |
| Inspect impact evidence | Scoped external JPL Sentry results and explicit conditional scenarios; not a validated local impact-risk solver |
| Recover batch work | Development review: owned REST and primary-menu SQL jobs, committed per-object checkpoints and restart recovery; other job families remain separate |
| Collect CAD candidates | Development only: one bounded Earth-encounter query, explicit partial coverage and optional object enrichment |

These capabilities have scoped engineering tests. They do **not** establish
independently calibrated artificial-origin probabilities, false-discovery rates or
five-sigma significance. A catalogue match, unusual spin or improved model fit does
not, by itself, establish natural, human-made or extraterrestrial origin.

## Install and run

Use Python **3.11**, a separate checkout and a new database. Do not install over a
dirty research tree or reuse an existing investigation database for initial tests.
These commands are for a POSIX shell on macOS/Linux:

```bash
git clone --branch v2.2.0b2 https://github.com/RobLe3/aneos-suite.git aneos-suite-beta
cd aneos-suite-beta
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
mkdir -p beta-runtime
export ANEOS_DATABASE_URL="sqlite:///$PWD/beta-runtime/aneos-beta.sqlite"
python aneos.py --version
python aneos.py help
```

The version command prints `aNEOS 2.2.0b2 (beta)`. Run commands from the checkout
root. Core dependencies include astronomy, numerical, plotting and API libraries;
this is not a dependency-free install. Optional ML packages do not confer
scientific validation. See the [installation guide](docs/user-guide/installation.md)
for isolation and troubleshooting.

### Screen one object — network required

```bash
python aneos.py analyze "1998 KY26" --json --output beta-runtime/ky26.json
```

This queries the configured providers, records acquisition outcomes and creates a
report. The output path must not exist; use a new filename for each assessment.
Providers may be slow, incomplete or unavailable. No particular score, completion
percentage or response time is promised. Add `--force-refresh` to bypass the
application source cache; that is not a guarantee against upstream caching.

For Markdown output, omit `--json`:

```bash
python aneos.py analyze "1998 KY26" --output beta-runtime/ky26.md
```

Each command above is a separate query, not a conversion of the earlier report.
Do not confuse acquisition failure with a successful natural-object result.
The CLI returns 0 for a report without recorded errors, 1 for a report containing
errors, and 2 when analysis/report creation fails. Exit 0 is not scientific approval.

### Read saved results without another query

```bash
python artificial_neo_dashboard_launcher.py --file beta-runtime/ky26.json
```

The file loader reads saved evidence; it does not acquire new observations. It
also creates an exclusive report bundle under `dashboard_results/` in the current
directory, so use a test directory when you do not want exports beside the checkout.
Graphics and tables do not add scientific significance. Keep the input file and
its provenance alongside exports. See the
[presentation contract](docs/engineering/polling-dashboard-screening.md).

### Use the menu

```bash
python aneos.py
```

Start with **1: Screen one object**, **2: Inspect screening evidence** and
**14: Read scientific help**. Option 9 browses results and option 10 exports them.
The [menu guide](docs/user-guide/menu-system.md) lists the actual option labels
and prerequisites. Not every legacy feature has completed behavioral QA.

### Start a local development API

```bash
python aneos.py api --dev --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the running application's generated schema.
The explicit loopback host matters: the underlying server otherwise defaults to
`0.0.0.0`. Do not expose the development server publicly. Authentication, operational
hardening and deployment readiness remain separate work; a successful Docker build
is not a production or PostgreSQL evidence-store qualification.

## Understand a result

Read the evidence and limitations before interpreting the rank:

| Result | Meaning | Does not mean |
|---|---|---|
| `FOLLOW_UP_CANDIDATE` | The installed ranking policy selected the object for further investigation | Confirmed artificial origin or discovery significance |
| `NOT_FLAGGED` | The current inputs did not pass that ranking policy | Proven natural origin, no interesting physics or no impact risk |
| `INCONCLUSIVE` | Evidence or computation is insufficient for that assessment | Zero anomaly or a negative scientific finding |

The current threshold is **5.0 ranking units**, not five Gaussian standard
deviations. Fixed population assumptions, correlated size/photometry features and
hand-set inclination bonuses remain limitations of the legacy rank. Conditional
reference populations and independent calibration are unfinished. Do not use the
score as an origin probability or treat contribution counts as independent evidence.
See the [screening contract](docs/engineering/screening-result-contract.md).

Origin probability is **not calibrated**; unsupported significance and FDR outputs
are unavailable. Physical-origin explanations require compatible observations,
uncertainties, competing force models and independent checks. They may remain
unresolved. aNEOS is not an operational planetary-defense warning service.

## Evidence and history

The installed named-object source profile includes SBDB, NEODyS, MPC, Horizons and
CAD. Actual availability and field coverage vary by provider and query. A successful
query does not imply all possible sources were searched or all measurements exist.
CAD encounters and Horizons ephemerides are computed predictions, not new independent
observations. Providers may also share underlying observations.

The existing local ledger retains source outcomes and accepted source-specific
evidence rather than replacing all history with one latest value. Raw payload
retention is path-specific; some adapters preserve parsed native JSON, while the
CAD collection preserves its buffered native response entity. Do not assume every
HTTP response byte or failed response body is retained.

- [Acquisition and field coverage](docs/engineering/object-acquisition-ledger.md)
- [Atomic physical measurements and uncertainty](docs/engineering/atomic-physical-evidence.md)
- [Durable evidence storage and export](docs/engineering/durable-enrichment-ledger.md)
- [Chronology and date roles](docs/engineering/object-chronology.md)

Source coverage, field presence and scientific fitness are different measurements.
Even 5/5 sources or 15/15 basic fields cannot establish complete scientific evidence.
Unavailable measurements must not be filled with zeros to reach 100%.

Hashes verify consistency with preserved content; they do not authenticate the
initial source or protect against a fully compromised provider/administrator.
Historical-integrity comparison is planned in
[issue #87](https://github.com/RobLe3/aneos-suite/issues/87), not a substantiated
finding of tampering. A modern query for an old epoch is not a historical capture.

## 1998 KY26 reference investigation

1998 KY26 is a regression and hypothesis-neutral case study, not a score-tuning
target. The [case register](docs/case-studies/1998-KY26/baseline.md) preserves the
recorded historical display **4.142**. Frozen rounded inputs reproduce
**4.141288008229254**, not the unknown full-precision original calculation.
Missing original provenance remains explicit.

The investigation must compare natural forces, measurement/model artifacts and
specific human-technological hypotheses using compatible evidence. It need not
support artificial origin to be useful. Observational model fitting, complete
historical reconciliation and independent scientific validation remain unfinished.

## Development and verification

From the installed development checkout, run offline regression tests:

```bash
python -m pytest tests/ aneos_core/tests/ -m "not network" --cov=aneos_core --cov-fail-under=30
```

CI also enforces component-specific coverage gates and builds the Docker image.
Tests use explicitly synthetic controls where appropriate. Passing them does not
establish real-world detection accuracy, calibration or exhaustive menu readiness.
Optional network checks are separately marked and require provider-safe access.

See [CONTRIBUTING.md](CONTRIBUTING.md), [ADRs](docs/architecture/ADR.md),
[DDD](docs/architecture/DDD.md) and the
[discovery/integrity roadmap review](docs/engineering/discovery-integrity-roadmap-review-2026-11-13.md).
The review reuses existing boundaries; it does not introduce a parallel detector or
database. Short-term work prioritizes correctness and preservation, followed by
qualified diagnostics/physical comparison, then independent prospective validation.

## Documentation and reporting problems

[Start with the documentation index](docs/README.md). Older design and scientific
pages may describe withdrawn or proposed capabilities; use the linked current
contracts to interpret them. The previous entry guides are preserved in the
[dated archive](docs/archive/entry-guides-2026-11-13/README.md), not current instructions.
The [documentation review](docs/engineering/entry-documentation-review-2026-11-13.md)
records what was checked and what remains.

When filing an [issue](https://github.com/RobLe3/aneos-suite/issues), include the
commit/version, exact command or menu choice, provider outcomes, relevant receipt
or policy identifiers, expected result and actual output. Remove secrets and private
observations. Do not attach a live research database without checking its contents
and redistribution rights.

## License

Read [LICENSE](LICENSE), including its additional ethical-use clause, before reuse.
