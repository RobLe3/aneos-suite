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
- 2026-03-07: Phase 3 completion. Ground truth validation executed against 3 confirmed artificials + 20 real JPL natural NEOs.
  - Tesla Roadster (2018 A1): σ=5.76, classified artificial ✅
  - 2020 SO (Centaur): σ=6.97, classified artificial ✅
  - J002E3 (Apollo 12 S-IVB): σ=5.76, classified artificial ✅
  - All 20 natural NEOs: classified natural ✅
  - Internal consistency check on N=4 objects (2 artificial, 2 natural), LOOCV.
  - Reported sensitivity=1.00, specificity=1.00, F1=1.00, ROC-AUC=1.00 on these 4 objects.
  - **CRITICAL CAVEAT**: F1=1.000 is NOT a generalisation estimate. With N=4 and thresholds
    hand-tuned to the same objects, perfect separation is mathematically expected, not evidential.
    95% Wilson CI on F1: approximately [0.0, 1.0]. An independent held-out corpus of ≥50 confirmed
    artificials is required before these metrics carry any statistical meaning.
  - NOTE: Bayesian posterior (max ~3–4%) cannot reach the theoretical 99.99994% confidence claim from
    orbital+physical evidence alone; that level requires propulsion/manoeuvre observations. The σ≥5
    detection threshold is met; the posterior probability reflects the 0.1% assumed base rate.
  - Likelihood ratios are hardcoded constants (10/5/2 by sigma tier), NOT calibrated against data.
  - See docs/scientific/VALIDATION_INTEGRITY.md for the full independent audit.

## Detection Interpretation Framework

| Metric | Meaning | Bound |
|--------|---------|-------|
| σ level | Rarity under Granvik 2018 natural NEO null hypothesis | Unbounded above 0 |
| P(artificial) | Bayesian posterior; 0.1% prior (asserted); LR hardcoded | ~1–5% from orbital+physical |
| Classification | σ ≥ 5.0 → flagged for human review | p < 5.7×10⁻⁷ (uncorrected) |

### Precise sigma definition

```
σ_eff = sqrt(Z_a² + Z_e² + Z_i²) + low_inclination_bonus
```

- Z_a, Z_e, Z_i are z-scores against Granvik 2018 population moments (Table 1)
- Under the null, the base term follows χ²(3); p_analytical = 1 − CDF_χ²(σ_eff², 3)
- **The low-inclination bonus is a hand-tuned additive constant (max +2.0 σ)** that is
  NOT part of the χ²(3) null; it inflates σ beyond what the stated p-value implies
- Orbital elements are treated as independent; their real-world correlations (a–e–i)
  are ignored, making p-values optimistic
- Multiple-testing correction is NOT applied across the population screened

Smoking-gun evidence (course corrections, propulsion burns, specular radar) needed
to push posterior above 10%. "σ=5 detection" means "extremely unusual under the null",
not "probably artificial".
