# aNEOS Validation Integrity Audit

**Date:** 2026-03-08
**Scope:** Detection pipeline — `ValidatedSigma5ArtificialNEODetector`
**Trigger:** Independent critique applying standard ML/statistics review criteria
**Verdict:** Proof-of-concept quality; not publication-grade; specific gaps documented below

---

## Overall Assessment

The software architecture is sound and the anomaly-ranking approach is scientifically
motivated. However, **the evaluation language is ahead of the evidence**. The reported
perfect metrics (F1=1.000, sensitivity=1.000, specificity=1.000) are artifacts of a
4-object internal consistency check, not evidence of generalisation. Five critical issues
must be resolved before any detection claim can be taken literally.

---

## 1. Ground Truth Corpus

### Current state

| Class | N | Source | Label basis |
|-------|---|--------|-------------|
| Artificial | 2 (CV) / 9 (full) | Hardcoded in detector / `ground_truth_dataset_preparation.py` | Confirmed: Tesla Roadster (SpaceX), DSCOVR (NOAA), 2020 SO (Surveyor Centaur), J002E3 (Apollo 12 S-IVB), plus 5 deep-space probes |
| Natural (CV) | 2 | Hardcoded in detector | Apophis + Bennu |
| Natural (external) | ≤250 | JPL SBDB Query API | `NEO=Y, e<0.99, H<26, q<1.3` filter |
| Natural (fallback) | 1,000 | Granvik 2018 synthetic, `seed=42` | Simulated orbital distribution |

### Issues

- **The cross-validation runs on only 4 objects** (2 per class), hardcoded in
  `_load_artificial_objects_database()` and `_load_natural_objects_database()`.
- **Thresholds were hand-tuned to these exact 4 objects.** The low-inclination bonus
  (≤+2.0 σ) is calibrated to Tesla Roadster (i=1.077°) and DSCOVR (i=0.1°). The mass
  anomaly score is calibrated to spacecraft masses (~570–1350 kg vs ~10¹¹ kg natural).
- **Self-referential validation:** When JPL APIs are unavailable, all artificial objects
  use hardcoded fallback orbital elements — the same values used to tune the scoring
  functions. The validation loop is testing the detector's own priors.
- The Granvik 2018 synthetic naturals (seed=42) reproduce the population model; they are
  not independent observational data.

### Minimum required to fix

≥50 confirmed artificial objects (or proxy objects with documented human-made origin)
and ≥200 independent natural NEOs in a held-out dataset that was not used during any
development phase, including threshold selection.

---

## 2. Train/Test Leakage

### Current state

The main detector does **not** split data in the conventional sense. Instead:

1. Global population parameters (Granvik 2018 moments) are fixed constants — not refit per fold.
2. The LOOCV runs `analyze_neo_validated()` on each held-out object, which calls
   `_bayesian_evidence_fusion()` with hardcoded likelihood ratios and priors.
3. **All threshold multipliers were chosen to make Tesla and DSCOVR score above 5σ and
   Apophis/Bennu score below 3σ.** No ablation study or sensitivity analysis documents
   how those multipliers were selected.

### Specific leakage points

| Parameter | Value | Justified? |
|-----------|-------|------------|
| Low-inclination bonus multiplier | 0.3 (strong), 0.1 (moderate) | No — hand-set to Tesla |
| Inclination bonus caps | 2.0, 0.5 | No — hand-set to produce σ≥5 for Tesla |
| Mass anomaly threshold | mass < 0.1× expected | No — calibrated to spacecraft masses |
| Likelihood ratio at σ>4 | 10 | No — asserted constant |
| Prior P(artificial) | 0.001 | No — asserted |

### Impact

Any trained threshold that was selected to pass specific objects and fail others, then
tested on those same objects, will produce artificially perfect metrics. This is
circular reasoning, not validation.

---

## 3. Statistical Meaning of Sigma

### Formula

```
σ_eff = sqrt(Z_a² + Z_e² + Z_i²) + low_inclination_bonus
```

where Z_x = (x − μ_x) / σ_x using Granvik 2018 population moments.

### Null distribution claimed

`p = 1 − CDF_χ²(σ_eff², 3)` — analytical, from `scipy.stats.chi2.cdf`

### Problems

1. **The bonus is outside the null model.** The bonus of up to +2.0 σ is an additive
   constant applied after the χ²(3) score is computed. A σ_eff value that includes this
   bonus does NOT have p = 1 − CDF_χ²(σ_eff², 3). The p-value is therefore too small
   (overstated significance) whenever the bonus is applied.

2. **Independence assumption.** The χ²(3) distribution requires the three z-scores to be
   independent standard normals under the null. Real NEO orbital elements (a, e, i) are
   correlated (e.g., Aten/Apollo/Amor families cluster). Ignoring correlations makes
   p-values optimistic.

3. **Non-Gaussianity.** The marginal distributions of a, e, i in the real NEO population
   are not Gaussian (e is bounded to [0,1]; i follows a roughly half-normal distribution
   in the low-inclination regime). Using Gaussian z-scores against skewed distributions
   inflates significance in the tails.

4. **No multiple-testing correction.** If 1,000 objects are screened, ~1 object will
   exceed p<0.001 by chance. The `statistical_testing.py` module implements Bonferroni
   and Benjamini-Hochberg corrections but they are not called from the detector.

### What σ=6.97 actually means for 2020 SO

Under the stated formula with bonus applied, it means: "this object's orbital elements
are at least 5+ standard deviations away from the Granvik mean in a 3D Euclidean sense,
plus a hand-tuned low-inclination bonus, using an independent-Gaussian null model." It
does not mean "the probability of a natural object having these elements is 1 in 10^12".

---

## 4. Bayesian Posterior Calibration

### Current implementation

```python
base_prior = 0.001  # 0.1% (asserted)
# Prior adjusted by signature_evidence_strength (capped at 0.5)
# Likelihood ratio: hardcoded 10 / 5 / 2 by sigma tier
posterior = LR * prior / (LR * prior + (1 − prior))
```

At σ>4, LR=10, prior=0.001:
`posterior = 10 × 0.001 / (10 × 0.001 + 0.999) ≈ 0.0099` (i.e., ~1%)

This is **not a calibrated probability**. A reliable posterior requires:
- Prior derived from the observed rate of confirmed artificials in NEO catalogs
- Likelihood ratios computed from P(sigma_bin | known artificial) / P(sigma_bin | known natural)
  using a training split of the ground truth corpus
- Calibration verified via a reliability diagram on a holdout set
- Brier score reported

**The fact that the Tesla Roadster — a known, confirmed spacecraft — receives
P(artificial)=0.037 is the clearest sign of miscalibration.** A well-calibrated
system should assign confirmed artificials a probability approaching 1.0 (or at least
>>0.5) given sufficient evidence. The posterior ceiling of ~3–5% reflects the
mathematical structure of the formula, not a meaningful probability estimate.

### To fix

Replace hardcoded LR with empirically computed frequency ratios from training data:
```python
lr = histogram_artificial[sigma_bin] / histogram_natural[sigma_bin]
```
Then run isotonic regression or Platt scaling on the validation fold to calibrate the
output against actual outcomes.

---

## 5. Real-World Holdout and Ground Truth Realism

### Current evaluation layers

| Layer | Description | Status |
|-------|-------------|--------|
| Toy set | 4 hardcoded objects | Used for CV — circular |
| Research set | 9 hardcoded artificials + ≤250 SBDB naturals | External validator (`ground_truth_dataset_preparation.py`) |
| Operational shadow set | Live NEO stream, no curation | **Not implemented** |
| Blind holdout | Frozen, unseen during development | **Not implemented** |

The external validator (`GroundTruthValidator`) achieves sensitivity=1.0 on the 9 known
artificials, but these objects have large, obvious signatures (spacecraft mass vs 10¹¹ kg
natural masses; very low inclinations). They are easy positives. The system has never
been tested on:
- Ambiguous objects with partial data
- Natural objects with genuinely unusual orbits (e.g., high-e near-cometary NEOs)
- Newly discovered objects before orbital refinement
- Objects where the API returns missing or inconsistent fields

---

## 6. Sigma Claims vs. Posterior Mismatch

The output displays both σ=6.97 and P(artificial)=0.037 for the same object. These
numbers are not contradictory, but they will be misread by most users:

- σ=6.97 sounds like "6.97 standard deviations above chance" → sounds like certainty
- P(artificial)=0.037 sounds like "3.7% probability" → sounds like uncertainty

The correct interpretation is: "this object's orbital geometry is very unusual under
a natural NEO null model (σ=6.97), but given a prior rate of 0.1% artificial objects,
the posterior belief in artificial origin is only ~3.7%". This is mathematically
consistent but the σ number misleads unless the sigma semantics are clearly communicated
at every display point.

---

## Remediation Roadmap

### Phase A — Immediate (no external data needed)

- [x] Add honest docstring to detector (done 2026-03-08)
- [x] Add methodology_note and 95% Wilson CIs to validation report (done 2026-03-08)
- [x] Fix cross-validation log line to say "internal consistency check, N=4" (done 2026-03-08)
- [x] Update sigma5_success_criteria.md with precise sigma formula (done 2026-03-08)
- [x] Create this document (done 2026-03-08)
- [ ] Apply multiple-testing Bonferroni correction when screening populations
      (code exists in `validation/statistical_testing.py`; needs wiring into detector)
- [ ] Add sigma semantics disclaimer to every UI display of σ values

### Phase B — Requires external data (~50+ confirmed artificials)

- [ ] Replace hardcoded LR with empirical frequency ratios from training split
- [ ] Run Platt scaling or isotonic regression for posterior calibration
- [ ] Generate reliability diagram and Brier score
- [ ] Run 5-fold stratified CV with proper grouping by object identity
- [ ] Report per-fold variance, not just mean metrics
- [ ] Validate on a frozen holdout set set aside before any development began

### Phase C — Operational validation

- [ ] Shadow-run on real JPL SBDB new-discovery stream for ≥90 days
- [ ] Track false-positive rate on confirmed-natural recent discoveries
- [ ] Temporal robustness: replay classification as orbital arcs grow
- [ ] Test on objects with missing or noisy field values

---

## Summary Table

| Critique Point | Status | Evidence |
|----------------|--------|----------|
| Label validity | Partial | 9 confirmed artificials; labels are correct but N is tiny |
| Leakage (threshold tuning) | **CONFIRMED** | LR/prior/bonus all hand-set on same objects used in CV |
| F1=1.000 validity | **SPURIOUS** | N=4, circular |
| Sigma null model correctness | **APPROXIMATE** | χ²(3) violated by bonus + correlations + non-Gaussianity |
| Multiple-testing correction | **MISSING** | Code exists, not called |
| Bayesian prior data-derived | **NO** | Asserted 0.001 |
| Likelihood ratio calibrated | **NO** | Hardcoded constants |
| Posterior calibration check | **NO** | No reliability diagram or Brier score |
| External holdout set | **NONE** | All evaluation on development objects |
| Temporal robustness | **UNTESTED** | No arc-replay study |
