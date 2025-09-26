# Updating the GitHub Repository

This checklist describes how to publish a fresh round of ANEOS work to GitHub now that the stabilization fixes are in place.

## 1. Sync and Branch
1. Fetch the latest default branch:
   ```bash
   git fetch origin main
   ```
2. Create a topic branch for the change set (replace the placeholder with a descriptive slug):
   ```bash
   git checkout -b fix/<short-summary> origin/main
   ```

## 2. Apply the Agreed Remediations
- Follow the remediation roadmap captured in `docs/engineering/maturity_assessment.md` and the configuration/detector actions under `docs/engineering/sigma5_success_criteria.md`.
- Implement the immediate mitigations and regression fixes identified during the Phase 0–5 reviews (cache/config parity, alert imports, AsyncMock import, model reload tests, etc.).

## 3. Validate Locally
1. Install dependencies (if not already present):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full regression suite:
   ```bash
   pytest
   ```
3. Capture the command output (keep the log for the PR description).

## 4. Stage and Commit
1. Stage the updated files:
   ```bash
   git add <files>
   ```
2. Create a descriptive commit message that ties back to the remediation plan:
   ```bash
   git commit -m "Restore cache/config compatibility and green the suite"
   ```

## 5. Publish and Open a Pull Request
1. Push the branch to GitHub:
   ```bash
   git push -u origin fix/<short-summary>
   ```
2. Open a pull request against `main` that summarizes:
   - What was fixed (import corrections, detector regressions, configuration parity).
   - How it was validated (`pytest`, targeted integration tests).
   - Outstanding follow-up items (e.g., external API health checks, Monte Carlo runs).

## 6. Post-Merge Tasks
- Tag the release if it represents a milestone (e.g., `git tag v0.7.1 && git push origin v0.7.1`).
- Update public documentation (README, CHANGELOG, release notes) to reflect the newly verified state.
- Monitor CI and alerting dashboards to confirm the deployment behaves as expected.

By adhering to these steps, every update keeps the codebase aligned with the sigma-5 detection objectives and maintains demonstrable test coverage.
