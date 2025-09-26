# aNEOS 0.7.0 Maturity Assessment

_Last updated: August 2025_

## Overview
The 0.7 stabilization release restored cross-project version alignment and
recovered the automated regression suite, but several production-readiness
criteria remain unmet. This note records the current evidence so future phases
stay anchored to the sigma-5 mission.

## Verification Snapshot
- **Automated Tests** – 61 unit and integration tests pass, covering cache,
  configuration, sigma-5 detection, model persistence, and the phase 4
  integration harness.
- **Manual Checks** – Invoking the historical polling workflow without optional
  pipeline components triggers the fallback simulation path, demonstrating that
  external API integrations are still unverified in this environment.

## Key Risks
1. **External Dependencies** – NASA/ESA data source integrations are not
   exercised automatically; missing components cause the system to simulate
   results instead of failing fast.
2. **Operational Proof** – README quick-start flows launch, yet no recent
   evidence shows sigma-5 detections confirmed against real catalogues.
3. **Documentation Drift** – Several public documents still advertise "production
   ready" status despite the open risks, creating mismatched expectations.

## Recommended Actions
1. **Integration Restoration** – Ship health checks for SBDB, NEODyS, MPC, and
   optional pipeline packages so initialization fails loudly when dependencies
   are absent.
2. **Operational Runs** – Execute and document end-to-end analyses against
   historical data once integrations are healthy to produce reproducible
   sigma-5 evidence packages.
3. **Documentation Refresh** – Update README, roadmap, and deployment guides as
   hardening work completes so messaging tracks observed capability.

## Alignment with Action Plan
These recommendations map directly onto the sequential adaptation–correction–
verification plan: Phase 1 covered baseline documentation/test repairs; the
next milestones focus on configuration parity (Phase 2), detector calibration
(Phase 3), integration proof (Phase 4), and production guardrails (Phase 5).
