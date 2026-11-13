# Dated Sentry lookup, separate from the historical detector baseline

One guarded live request for primary designation `1998 KY26` completed at
**2026-11-12T21:19:13.177037Z**. Sentry API 2.0 returned `specified object not found`.
The report therefore says **not listed**, with null impact probability and
not-assessed risk. This is not zero risk, a natural-origin result or a refutation
of any origin hypothesis. It supplies no local observational fit or acceleration.

- `raw-response.json`: exact response bytes. SHA-256:
  `722bec8de705159e59e00f6be8592e1233b0301370c65a786e2a0072cfeba2b9`.
- `report.json` / `report.md`: replayable machine/human evidence; digest
  `7b7de1592593fc476c895b2ad429f7bb8db05f306640d2f7242be40ea15fe947`.
- `execution.json`: base revision and acquisition-source file hashes. Capture ran
  on the isolated remediation worktree before its new implementation commit;
  the listed source file hashes identify the exact then-uncommitted adapter code.
  This metadata is not a complete dependency lock or full environment reproduction.

Only this short public API lookup response is retained here, not papers or source
observational archives. Replay is offline and tested against the stored bytes.
Hashes detect changed content, not provenance fraud or cryptographic source identity.
See the [primary API contract](https://ssd-api.jpl.nasa.gov/doc/sentry.html).
The original November 12 detector baseline and the separate SBDB measurement
capture are unchanged. Future queries must create new dated records, not replace
these files. This directory is an example frozen record, not a completed durable
case-lifecycle implementation (#13).

`acquisition-code/*.txt` archives those three exact source files by basename. The
current adapter additionally handles fractional-day observation dates exposed by a
separate available-MC source control; the archived code describes this original
KY26 acquisition, not the final corrected adapter. Current replay of these same
KY26 bytes remains identical. No historical source hash has been silently replaced.
