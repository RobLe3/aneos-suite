# Analysis policy: inspect, diagnose and archive

Development review, 2026-11-13; #56/#57, ADR085. The legacy Analysis Policy menu now
reports what the default canonical screening service actually does. It does not apply
scientific settings that lack a supported runtime contract.

## What the menu means

The default follow-up threshold remains **5 raw ranking units**, unchanged from the
historical screening policy. It is not five Gaussian sigma or an origin-confidence
threshold. The policy fingerprint includes its version, threshold, engine version and
fixed reference parameters. The service now exposes the same description used to compute
assessment policy IDs; inspection does not create a second scientific engine.

The current reference parameters remain uncalibrated. Size/photometry dependence,
manual inclination contributions, population selection and measurement uncertainty still
need scientific work. FOLLOW_UP_CANDIDATE means review is warranted under the ranking
policy; NOT_FLAGGED does not establish natural origin. Probability, significance and
reliable FDR are unavailable. No score was changed for KY26.

The view describes a newly constructed **default canonical service**. An explicitly
customized service or an already running job may use a different recorded policy ID.
It is not a live audit of all workers or entry points. Required-review-v1 remains a
separate mandatory workflow contract; opening this menu does not run or certify it.

## Navigation and actions

The legacy grouped menu (`python aneos.py --legacy-menu`) routes NEO Detection &
Analysis → **9: Analysis Policy** to this view. Professional Mode's existing C action
also retains the compatibility entry. Tests exercise the actual detection-submenu
route and the action handlers; they do not certify every legacy startup prerequisite.

| Action | Actual behavior |
| --- | --- |
| 1: Significance controls | Explains why alpha changes are not applied; no calibrated null/search-wide procedure is installed. |
| 2: Physical model comparison | Explains legacy BIC limitations; no validated physical-fit toggle is applied. |
| 3: Follow-up threshold | Shows the actual default policy; does not claim a threshold change. |
| 4: Synthetic controls | Runs positive, ordinary and missing-input cases through the real canonical service. These are not observations or independent validation. |
| 5: Export policy snapshot | Writes a new JSON record to an explicit path; refuses existing files/symlinks. Blank cancels. |
| 6: Verify policy snapshot | Checks structure/checksums, then reports matches/differs from current default. Never imports or applies settings. Blank cancels. |
| 0: Return | No changes. |

Legacy helper names remain available, with unsupported mutations explicitly labelled
not applied. There is no “thresholds updated” or demonstration-import success. If a
prerequisite fails, the menu reports unavailable without exposing provider/private
exception text or manufacturing a scientific assessment. Both Rich and plain terminals
are supported. The launcher no longer advertises an unaudited121-option coverage count.

## Inspect the policy programmatically

This example was executed against the current implementation. It performs no source
polling, database write or runtime policy mutation:

```python
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.analysis.policy_inspection import build_policy_snapshot, verify_policy_snapshot

service = ScreeningAnalysisService()
snapshot = build_policy_snapshot(service)
assessment = service.analyze_raw('SYNTHETIC', {'a': 1.2, 'e': 0.2, 'i': 20.0})
assert snapshot['policy_description']['policy_id'] == assessment.policy_id
comparison = verify_policy_snapshot(snapshot, service=service)
assert comparison['comparison'] == 'matches_active'
assert comparison['runtime_configuration_changed'] is False
```

The `screening-policy-inspection-v1` record contains application version, an explicit
UTC inspection timestamp, actual policy/reference description, declared limitations and
a snapshot hash. It is not an ANEOSConfig file or an executable runtime policy package.

Verification rejects malformed/oversized/nonfinite JSON, duplicate keys, unsupported
schemas/capability claims, bad timestamps, invalid threshold domains and inconsistent
IDs. A self-consistent archive can differ from the active default; this is reported, not
silently installed. Checksums establish consistency, **not authenticity**. An author can
recompute them; no signature, calibration or administrative security is implied.

Export uses exclusive creation, flush and fsync. Existing files are not overwritten.
An I/O failure can leave a partial **new** file; the error says export was not completed
and does not delete it automatically. This is not a backup or a universal power-loss
promise. General `ANEOSConfig.save()` remains a separate overwriting API; protect an
existing configuration before calling that method.

## General configuration corrections

Disposable-file/runtime tests reproduced two independent defects: malformed explicit
files silently activated defaults, and rejected updates left invalid values installed.
The existing loader now raises for explicit missing/malformed/unknown-key files. A valid
explicit file takes precedence over the environment; there is no environment-over-file
merge. Without an explicit file, supported environment variables overlay defaults.

ConfigManager updates/reloads validate a candidate before replacing that manager's
object. Rejected operations preserve the previous object and values; valid updates do
not mutate a previously returned object or persist changes to disk. Unknown update
fields fail. Checked eccentricity/inclination, timeout/retry and worker values have
strict finite/type/domain checks. Other configuration fields and actual consumers still
need their own validation; do not infer whole-system guarantees from those checks.
Direct accessor objects remain mutable, and concurrent-manager/process coordination is
not implemented here. An in-memory update does not hot-reconfigure existing services.
See the corrected [configuration reference](../reference/configuration-reference.md).

## Reproduction and remaining work

```sh
python -m pytest tests/test_policy_configuration_menu.py -q --tb=short \
  --cov=aneos_core --cov-config=tests/coverage-policy-inspection.ini \
  --cov-report=term-missing --cov-fail-under=95
python scripts/qa_policy_menu.py --output /tmp/aneos-policy-new-capture
```

The capture generator refuses an existing directory. It freezes only its inspection
timestamp, uses the actual default policy, runs actual synthetic controls, and injects
one missing-prerequisite case. Text/HTML/SVG at80/120/160 columns plus the policy record
and source/artifact hashes are in `docs/qa/policy-menu/2.2.0.dev0/`. A fresh-process test
verifies an exported record; it does not claim runtime settings persist or are applied.

Three original menu regressions and five loader regressions failed before repair.
Focused/full-suite and exact-head CI measurements are recorded on the linked draft PR.
Existing quality gates remain intact, with an additional95% policy-inspection module
gate. This is not all-menu or all-field certification. #40 all-source query persistence,
#41 every menu, #43 observational models/calibration, #47 broader legacy auth and #3 exact
original baseline remain release blockers. No final KY26 origin assessment is produced.

## Claim/evidence check

| Earlier claim or impression | Evidence and correction | Scope still unresolved |
| --- | --- | --- |
| The configuration view provides99.99994% discovery confidence and natural/artificial thresholds. | Original actual-menu regression failed; canonical outputs have unavailable significance/origin. The repaired view labels raw ranking and shows the real policy ID. | Independent null/origin calibration and search-wide error control are not installed. |
| Alpha/threshold edits or import apply runtime settings. | Original helper changed only a throwaway dictionary; threshold/import paths had no application. They now explicitly report not applied, and snapshot verification is read-only. | A supported, audited scientific-policy deployment system is future work, not simulated here. |
| Environment values override an explicitly loaded file. | A disposable file with timeout7 versus environment99 selects7. The reference now matches the actual file-or-environment selection. | Every provider/worker honoring every configuration field is not established. |
| A rejected manager update leaves the prior valid configuration active. | The original regression showed max_workers=-1 surviving a raised error. Candidate-copy validation now preserves the previous manager object; reload failures are also tested. | Direct accessor mutation and concurrent multi-process configuration coordination are not protected. |

The immediate benefit is less misleading output and preserved prior state on rejected
operations. The systemic benefit is clearer separation of inspection, configuration and
scientific evidence. Long-term scientific credibility still depends on the unresolved
observational and calibration work, not menu polish. No intent or motive is inferred
from the old defects; limited tests are not independent certification.

### Measured delivery checkpoint

- Full offline suite: **1344 passed, 8 network tests excluded**; core coverage48.46%
  against the unchanged30% gate. These are software tests, not detection performance.
- Focused policy/configuration suite: **80 passed**; policy-inspection module100%
  coverage against the new95% gate. Existing component gates remain unchanged.
- Combined new tests plus existing configuration/canonical/legacy-menu regressions:
  **169 passed**. Original red runs contained three menu failures and five loader failures.
- Actual Rich inspection has39/33/32lines at80/120/160 columns; diagnostic and
  unavailable-prerequisite views also stay within the declared widths. Source/artifact
  hashes match, raster previews were visually inspected, and the Python example ran.
- No live provider query, new KY26 score, physical fit, calibrated probability or
  independent evaluation is part of this block. Historical ranking policy identity and
  baseline tests remain unchanged; source/observational/calibration gaps stay open.
- Changed implementation: canonical policy description, policy inspection module,
  existing menu helpers/routing labels, bounded general configuration load/update safety
  and launcher help. ADR085/DDD, configuration reference, guide, README/changelog,
  tests/CI and versioned capture artifacts document the changes. Exact reviewed commit
  and CI outcome are recorded on the draft PR; no issue closes automatically.
