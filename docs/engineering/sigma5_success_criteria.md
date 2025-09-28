# Sigma-5 Detection Success Criteria (Phase 0 Alignment)

This note records the success criteria implied by the public documentation so that all remediation work in subsequent phases
remains aligned with the published aNEOS goal of a multi-modal σ≥5 detector.

## Source of Truth
The criteria below are transcribed directly from the project README, which currently defines the external expectations for the
system's capabilities. Any future change to the README must be reflected here to keep the engineering guardrails current.

## Target Outcome
- **Sigma Threshold**: The detector must achieve at least a 5-sigma composite score when classifying an object as artificial,
  matching the "Multi-Modal Sigma 5 Detection" claim.
- **Statistical Certainty**: Deliver 99.99994% confidence (the statistical meaning of a 5-sigma result) for artificial
  classifications.
- **False-Positive Ceiling**: Maintain a false-positive rate below 5.7×10⁻⁷ (approximately 1 in 1.74 million) as advertised in
  the README.
- **Validation Requirement**: Develop validated exemplars using confirmed artificial objects (NOTE: Tesla Roadster claim was removed as no such validation exists in current documentation).

## Verification Log
- 2025-09-25: README review performed during Phase 0 alignment; criteria captured verbatim for ongoing reference.
- 2025-09-25: Phase 0 revisit confirmed README language is unchanged; proceeding to Phase 1 remediation using these criteria.
