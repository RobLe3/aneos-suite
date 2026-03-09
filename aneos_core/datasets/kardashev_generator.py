"""
Kardashev Synthetic NEO Generator

Generates physically plausible artificial NEO signatures across Kardashev
civilisation levels 0.5–2.0 for use as training positives in the aNEOS ML
pipeline.

SCIENTIFIC BASIS
----------------
Each Kardashev tier implies specific engineering capabilities and constraints,
which translate into observable orbital and physical properties:

  K0.5  Near-planetary  (~10^15 W)   Chemical/ion rockets, solar sails
  K1.0  Planetary       (~10^17 W)   Fusion drives, automated megaprojects
  K1.5  Stellar fringe  (~10^19 W)   Partial Dyson swarm, directed energy
  K2.0  Full stellar    (~10^26 W)   Complete stellar engineering

For each tier, 3–4 sub-scenarios (named "combos") are defined.  Each scenario
specifies probability distributions over the features our current sensor suite
can actually measure:

  Orbital:   a (AU), e, i (°), ω (°), Ω (°), M (°)
  Physical:  diameter_km, albedo, density_g_cm3, rotation_period_hours
  Non-grav:  A1 (radial), A2 (transverse), A3 (normal) [AU/day²]

KEY INSIGHT — inversion from natural training
---------------------------------------------
Natural NEO distributions are well-sampled (~35 000 objects in SBDB).  The
natural manifold in (a, e, i, albedo, density) space can be learned from real
data.  Anything that lands in a low-probability region of that manifold is
anomalous.  The Kardashev scenarios define *where* in that anomaly space we
expect artificials to cluster, allowing a supervised or semi-supervised
classifier to associate specific anomaly signatures with specific scenarios.

PARAMETER DISTRIBUTION CHOICES
--------------------------------
  a         TruncatedNormal or Uniform — bounded to physical range
  e         Beta(α,β)  — naturally in [0,1]; tuned per tier
  i         TruncatedNormal(μ=ecliptic_bias, σ=spread) or Uniform(0,90)
  albedo    Beta(α,β) or bimodal mix (solar panel vs metallic surface)
  density   LogNormal(μ_log, σ_log) — always positive, spans orders of magnitude
  A2        LogNormal(μ_log, σ_log) — non-grav transverse, sign flipped randomly
  rotation  LogNormal — ranges from fast spin to tumbling

LIMITATIONS
-----------
Scenario parameters are derived from physics, not empirical artificial-NEO
observations.  A model trained on this corpus may not generalise to truly
alien engineering.  Treat as an informed prior, not ground truth.  Label all
records with source=synthetic_kardashev so they can be down-weighted.

REFERENCE NATURAL POPULATION (for context)
-------------------------------------------
  a:    0.5 – 4.0 AU  (NEA family-dependent)
  e:    0.0 – 0.8     (typical 0.2 – 0.5)
  i:    0° – 60°      (typical 5° – 30°)
  albedo:  0.04 – 0.6  (typical S/C class ~0.2)
  density: 1.2 – 4.0 g/cm³
  A2:   effectively 0 (Yarkovsky is ~1e-15 to 1e-13 AU/day²)
  rotation: 2 – 1000 h  (most < 100 h)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Natural-population reference bounds (used for anomaly-score sanity checks)
# ---------------------------------------------------------------------------

_NATURAL_A_RANGE       = (0.5, 4.0)   # AU
_NATURAL_E_MAX         = 0.9
_NATURAL_I_MAX_DEG     = 60.0
_NATURAL_ALBEDO_RANGE  = (0.04, 0.6)
_NATURAL_DENSITY_RANGE = (1.2, 4.0)   # g/cm³
_NATURAL_A2_MAX        = 5e-13        # AU/day² — strong Yarkovsky upper bound


# ===========================================================================
# Tier and scenario definitions
# ===========================================================================

class KardashevTier(Enum):
    """Kardashev scale tier labels."""
    K0_5 = "K0.5"   # Near-planetary (~10^15 W)
    K1_0 = "K1.0"   # Planetary      (~10^17 W)
    K1_5 = "K1.5"   # Stellar fringe (~10^19 W)
    K2_0 = "K2.0"   # Full stellar   (~10^26 W)


@dataclass
class ScenarioSpec:
    """
    Complete specification of a single Kardashev scenario ("combo").

    Each field is a (mean, std) pair for a TruncatedNormal sample, or a
    (low, high) pair for a Uniform sample, or a (mu_log, sigma_log) pair
    for a LogNormal sample.  The 'kind' field on the sub-dataclass
    distinguishes these.

    Non-grav magnitudes are given as 10-log (log10) of the absolute value;
    sign is randomised independently.
    """
    tier: KardashevTier
    name: str
    description: str

    # --- orbital element distributions ---
    # (low, high) for Uniform; (mean, std) for truncnorm
    a_dist:   Tuple[float, float]   # AU; semi-major axis
    e_dist:   Tuple[float, float]   # [0,1]; eccentricity
    i_dist:   Tuple[float, float]   # degrees; inclination
    # Ω and ω are typically uniform for artificial objects (no preferred plane)
    # unless stated otherwise — we draw uniform [0, 360] by default.

    # --- physical property distributions (log-normal: mu_log10, sigma_log10) ---
    diameter_log_dist:  Tuple[float, float]  # log10(km)
    albedo_dist:        Tuple[float, float]  # direct Beta params (alpha, beta)
    density_log_dist:   Tuple[float, float]  # log10(g/cm³)
    rotation_log_dist:  Tuple[float, float]  # log10(hours)

    # --- non-grav transverse A2 (most detectable): log10(|A2|) in AU/day² ---
    a2_log_dist: Tuple[float, float]   # (mu_log10, sigma_log10)
    # A1 (radial component): often ~0.3–0.5× A2 for radiation pressure
    a1_fraction_of_a2: float = 0.3    # ratio A1/A2 (approximate)

    # --- anomaly fingerprint weight (0–1): how strongly each feature
    #     deviates from the natural population ---
    signature_strength: float = 0.8   # informational; not used in sampling

    # --- sampling mode for a and i ---
    a_mode: str = "uniform"   # "uniform" or "normal"
    i_mode: str = "uniform"   # "uniform" or "normal"

    def sample(
        self,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """
        Draw one synthetic NEO from this scenario's distributions.
        Returns a flat dict of raw parameter values.
        """
        # --- orbital elements ---
        a = self._draw(rng, self.a_dist, self.a_mode,  0.01, 10.0)
        e = self._draw(rng, self.e_dist, "uniform",    0.0,  0.999)
        i = self._draw(rng, self.i_dist, self.i_mode,  0.0,  180.0)
        om = rng.uniform(0.0, 360.0)   # Ω: longitude of ascending node
        w  = rng.uniform(0.0, 360.0)   # ω: argument of periapsis
        M  = rng.uniform(0.0, 360.0)   # M: mean anomaly

        # --- physical properties ---
        diameter_km = 10 ** self._draw(rng, self.diameter_log_dist,
                                       "normal", -3.0, 4.0)
        alpha, beta_ = self.albedo_dist
        albedo = float(rng.beta(alpha, beta_))
        albedo = float(np.clip(albedo, 0.01, 0.99))

        density = 10 ** self._draw(rng, self.density_log_dist,
                                   "normal", -5.0, 1.5)

        rotation_h = 10 ** self._draw(rng, self.rotation_log_dist,
                                      "normal", -1.0, 5.0)

        # Absolute magnitude from diameter + albedo (IAU formula)
        h_mag = self._diameter_to_h(diameter_km, albedo)

        # --- non-gravitational parameters ---
        log_a2 = self._draw(rng, self.a2_log_dist, "normal", -16.0, -6.0)
        a2_mag = 10 ** log_a2
        a2_sign = rng.choice([-1.0, 1.0])
        A2 = a2_sign * a2_mag

        A1 = A2 * self.a1_fraction_of_a2 * rng.choice([-1.0, 1.0])
        A3 = 0.0  # out-of-plane: effectively undetectable for current surveys

        return {
            # Orbital
            "a":  a,
            "e":  e,
            "i":  i,
            "om": om,
            "w":  w,
            "M":  M,
            # Physical
            "diameter_km":           diameter_km,
            "albedo":                albedo,
            "density_g_cm3":         density,
            "rotation_period_hours": rotation_h,
            "absolute_magnitude_h":  h_mag,
            # Non-grav
            "nongrav_A1": A1,
            "nongrav_A2": A2,
            "nongrav_A3": A3,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw(rng: np.random.Generator, dist: Tuple[float, float],
              mode: str, lo: float, hi: float) -> float:
        """Draw a value from a uniform or truncated-normal distribution."""
        if mode == "uniform":
            return float(rng.uniform(dist[0], dist[1]))
        else:  # normal, truncated to [lo, hi]
            mu, sigma = dist
            for _ in range(20):
                v = float(rng.normal(mu, sigma))
                if lo <= v <= hi:
                    return v
            return float(np.clip(mu, lo, hi))

    @staticmethod
    def _diameter_to_h(d_km: float, albedo: float) -> float:
        """Compute absolute magnitude H from diameter (km) and albedo."""
        # H = -5 log10(d / (1329 * sqrt(pv)))
        if d_km <= 0 or albedo <= 0:
            return 25.0
        try:
            return -5.0 * math.log10(d_km / (1329.0 * math.sqrt(albedo)))
        except (ValueError, ZeroDivisionError):
            return 25.0


# ===========================================================================
# Scenario catalogue — the 14 "combos"
# ===========================================================================

def _build_scenario_catalogue() -> List[ScenarioSpec]:
    """
    Define all 14 Kardashev × sub-scenario combinations.

    Parameter rationale for each scenario is documented inline.
    """
    scenarios: List[ScenarioSpec] = []

    # -----------------------------------------------------------------------
    # K0.5 — Near-Planetary (~10^15 W)
    # Technology: chemical rockets, ion drives, solar sails, nuclear-electric
    # -----------------------------------------------------------------------

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K0_5,
        name="dead_rocket_stage",
        description=(
            "Defunct upper-stage rocket body: tumbling, high area-to-mass, "
            "mostly in Earth-crossing injection orbits.  Radiation pressure "
            "dominates non-grav.  A2 elevated ~10× Yarkovsky maximum."
        ),
        # Injection orbits: a ≈ 1.0–1.5 AU, moderate e from launch window
        a_dist=(0.85, 1.55),   a_mode="uniform",
        e_dist=(0.05, 0.50),
        i_dist=(0.0,  30.0),   i_mode="uniform",
        # Light hollow metal cylinder: 2–8 m effective diameter
        diameter_log_dist=(-2.0, 0.3),   # 10^-2 to 10^-0.5 km ≈ 10m–300m equiv
        albedo_dist=(5.0, 3.0),           # Beta(5,3) → mean ≈ 0.625 (metallic)
        density_log_dist=(-2.0, 0.4),     # ~0.001–0.05 g/cm³ (hollow)
        rotation_log_dist=(2.0, 0.6),     # ~100 h tumbling (log10)
        a2_log_dist=(-12.0, 0.8),         # |A2| ≈ 1e-13 to 1e-11 AU/day²
        a1_fraction_of_a2=0.25,
        signature_strength=0.75,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K0_5,
        name="solar_sail",
        description=(
            "Active or retired solar sail: enormous area-to-mass ratio, "
            "A2 >> any natural body.  Very low mass, diameter estimate huge "
            "but density near zero.  May have near-zero eccentricity if active."
        ),
        a_dist=(0.70, 1.30),   a_mode="uniform",
        e_dist=(0.001, 0.15),  # sails maintain low-e orbits or spiral inward
        i_dist=(0.0,  25.0),   i_mode="uniform",
        diameter_log_dist=(-1.0, 0.5),   # 0.1–3 km effective
        albedo_dist=(9.0, 1.5),           # Beta(9,1.5) → very high (mirror sail)
        density_log_dist=(-5.0, 0.5),     # ~1e-5 g/cm³
        rotation_log_dist=(3.0, 0.8),     # slow tumble or controlled
        a2_log_dist=(-10.5, 0.6),         # very large |A2|
        a1_fraction_of_a2=0.9,            # radial component significant too
        signature_strength=0.95,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K0_5,
        name="active_ion_probe",
        description=(
            "Active ion-drive probe: continuous low thrust produces slowly "
            "evolving a (Δa detectable over years in orbital history).  "
            "A2 magnitude at threshold of SBDB detectability."
        ),
        a_dist=(0.80, 2.20),   a_mode="uniform",
        e_dist=(0.02, 0.45),
        i_dist=(0.0,  45.0),   i_mode="uniform",
        diameter_log_dist=(-3.0, 0.4),   # small probe: ~1–10 m equiv diameter
        albedo_dist=(4.0, 2.0),           # moderate metallic
        density_log_dist=(-1.5, 0.4),
        rotation_log_dist=(1.5, 0.7),     # relatively slow but known
        a2_log_dist=(-12.5, 0.7),         # ion thrust ~1e-13 AU/day²
        a1_fraction_of_a2=0.1,
        signature_strength=0.70,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K0_5,
        name="mining_tug",
        description=(
            "Autonomous resource-extraction vehicle operating in the inner "
            "asteroid belt.  Heavier, lower A/m than sail.  Occasional "
            "thrust for rendezvous creates discontinuities in orbital history."
        ),
        a_dist=(1.0,  2.5),    a_mode="uniform",
        e_dist=(0.05, 0.60),
        i_dist=(0.0,  35.0),   i_mode="uniform",
        diameter_log_dist=(-2.5, 0.5),
        albedo_dist=(3.0, 2.0),
        density_log_dist=(-1.0, 0.5),     # ~0.01–0.2 g/cm³
        rotation_log_dist=(2.5, 0.7),
        a2_log_dist=(-13.0, 1.0),
        a1_fraction_of_a2=0.2,
        signature_strength=0.65,
    ))

    # -----------------------------------------------------------------------
    # K1.0 — Planetary (~10^17 W)
    # Technology: fusion/nuclear drives, automated megaprojects, full inner
    # solar system access, cheap plane changes
    # -----------------------------------------------------------------------

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_0,
        name="deep_space_vehicle",
        description=(
            "High-energy interplanetary vehicle: wide a range, significant "
            "A2 from active thrust.  Plane changes cheap → higher i than "
            "K0.5.  Discrete Δv events visible in orbital history."
        ),
        a_dist=(0.40, 4.0),    a_mode="uniform",
        e_dist=(0.02, 0.85),
        i_dist=(0.0,  65.0),   i_mode="uniform",
        diameter_log_dist=(-1.5, 0.6),
        albedo_dist=(4.0, 2.5),
        density_log_dist=(-2.5, 0.5),
        rotation_log_dist=(1.0, 0.8),
        a2_log_dist=(-11.0, 0.9),         # active thrust → large A2
        a1_fraction_of_a2=0.4,
        signature_strength=0.85,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_0,
        name="automated_constructor",
        description=(
            "Large automated construction platform parked in a stable orbital "
            "resonance with Jupiter (a ≈ 2.06, 2.5 or 3.28 AU; 3:1, 5:2, 2:1 "
            "MMR).  Very low eccentricity (deliberate).  Large but low density."
        ),
        # a biased toward Jupiter MMR locations
        a_dist=(1.90, 3.40),   a_mode="uniform",
        e_dist=(0.001, 0.08),  # near-circular
        i_dist=(0.0,  55.0),   i_mode="uniform",
        diameter_log_dist=(0.0, 0.5),     # 1–10 km effective
        albedo_dist=(6.0, 2.0),           # processed metal/glass
        density_log_dist=(-3.0, 0.5),     # very low density megastructure
        rotation_log_dist=(3.0, 1.0),
        a2_log_dist=(-12.0, 1.0),
        a1_fraction_of_a2=0.15,
        signature_strength=0.80,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_0,
        name="resource_platform",
        description=(
            "Deliberately circular orbit resource depot: very low e, "
            "moderate a, massive but ultra-low density (pressurised habitat "
            "or large storage vessel).  Essentially no detectable A2 when "
            "station-keeping is perfect, but slight imbalance visible."
        ),
        a_dist=(0.95, 2.0),    a_mode="normal",  # (mean, std)
        e_dist=(0.0001, 0.02),
        i_dist=(0.0,  50.0),   i_mode="uniform",
        diameter_log_dist=(0.5, 0.6),     # 3–100 km scale
        albedo_dist=(5.0, 3.0),
        density_log_dist=(-3.5, 0.5),
        rotation_log_dist=(4.0, 1.0),     # very slow; station-keeping
        a2_log_dist=(-13.5, 1.0),
        a1_fraction_of_a2=0.1,
        signature_strength=0.70,
    ))

    # -----------------------------------------------------------------------
    # K1.5 — Stellar Fringe (~10^19 W)
    # Technology: partial Dyson swarm, directed energy, interstellar probe
    # launches, stellar mining
    # -----------------------------------------------------------------------

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_5,
        name="dyson_swarm_element",
        description=(
            "Single element of a Dyson swarm: near-circular orbit very close "
            "to 1 AU (tuned to solar flux), enormously high area-to-mass.  "
            "Albedo near 1.0 (mirror collector), near-zero density, "
            "quasi-uniform distribution in inclination (3D swarm)."
        ),
        a_dist=(0.90, 1.10),   a_mode="uniform",
        e_dist=(0.0001, 0.008),  # extremely circular
        i_dist=(0.0,  180.0),  i_mode="uniform",  # 3D swarm
        diameter_log_dist=(0.0, 1.0),     # 1–1000 km effective
        albedo_dist=(18.0, 2.0),          # Beta near 1.0 (mirror)
        density_log_dist=(-5.5, 0.5),     # 1e-6 g/cm³ — essentially a sail
        rotation_log_dist=(4.5, 1.0),
        a2_log_dist=(-9.0, 0.8),          # enormous A2 from huge A/m
        a1_fraction_of_a2=1.0,            # radial component large too
        signature_strength=0.98,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_5,
        name="interstellar_launcher",
        description=(
            "Directed-energy interstellar launch platform: extremely high "
            "non-grav from recoil during beam-firing.  Near-Sun perihelion "
            "for maximum flux.  Highly eccentric orbit to gain velocity."
        ),
        a_dist=(1.0,  5.0),    a_mode="uniform",
        e_dist=(0.50, 0.98),   # high e → Sun-grazing perihelion
        i_dist=(0.0,  90.0),   i_mode="uniform",
        diameter_log_dist=(1.0, 0.8),
        albedo_dist=(8.0, 2.0),
        density_log_dist=(-4.0, 0.8),
        rotation_log_dist=(3.5, 1.0),
        a2_log_dist=(-8.5, 0.8),          # very large A2
        a1_fraction_of_a2=0.6,
        signature_strength=0.92,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_5,
        name="stellar_trawler",
        description=(
            "Highly eccentric object making deep perihelion passes for maximum "
            "solar energy collection.  Perihelion q < 0.2 AU; A1 (radial) is "
            "very large near perihelion.  Long-period orbit, very high e."
        ),
        a_dist=(2.0,  8.0),    a_mode="uniform",
        e_dist=(0.80, 0.97),
        i_dist=(0.0,  70.0),   i_mode="uniform",
        diameter_log_dist=(0.5, 0.8),
        albedo_dist=(7.0, 2.0),
        density_log_dist=(-3.5, 0.6),
        rotation_log_dist=(4.0, 1.0),
        a2_log_dist=(-9.5, 0.8),
        a1_fraction_of_a2=2.0,           # radial outshines transverse near perihelion
        signature_strength=0.88,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K1_5,
        name="population_coordinator",
        description=(
            "Coordinated swarm: individual element appears near-natural but "
            "is detectable only via population-level network sigma.  "
            "Tight inclination and node clustering; moderate a near resonances."
        ),
        a_dist=(1.0,  3.0),    a_mode="uniform",
        e_dist=(0.005, 0.10),
        # Clustering in i around a specific value — simulate by narrow normal
        i_dist=(25.0, 3.0),    i_mode="normal",
        diameter_log_dist=(-1.0, 0.5),
        albedo_dist=(5.0, 2.5),
        density_log_dist=(-4.0, 0.5),
        rotation_log_dist=(3.0, 0.8),
        a2_log_dist=(-11.5, 0.7),
        a1_fraction_of_a2=0.3,
        signature_strength=0.60,  # weak single-object signal; pop-level needed
    ))

    # -----------------------------------------------------------------------
    # K2.0 — Full Stellar (~10^26 W)
    # Technology: complete stellar energy utilisation, gravitational engineering
    # -----------------------------------------------------------------------

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K2_0,
        name="megastructure_fragment",
        description=(
            "Fragment of a star-scale megastructure: impossibly low density, "
            "vast but ultra-thin.  A2 exceeds solar radiation pressure by "
            "orders of magnitude — no known natural mechanism produces this."
        ),
        a_dist=(0.80, 1.30),   a_mode="uniform",
        e_dist=(0.0001, 0.05),
        i_dist=(0.0,  180.0),  i_mode="uniform",
        diameter_log_dist=(2.0, 1.0),    # 100 km – 10 000 km
        albedo_dist=(17.0, 2.0),         # near-perfect mirror
        density_log_dist=(-7.0, 0.5),    # ~1e-7 g/cm³
        rotation_log_dist=(5.0, 1.0),
        a2_log_dist=(-7.0, 0.8),         # |A2| ~1e-8 to 1e-6 AU/day²
        a1_fraction_of_a2=1.0,
        signature_strength=0.999,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K2_0,
        name="stellar_engine_component",
        description=(
            "Component of a stellar propulsion system (Shkadov thruster etc): "
            "persistent non-grav from stellar wind interaction far above "
            "natural levels; stable in theoretically chaotic regions."
        ),
        a_dist=(0.30, 6.0),    a_mode="uniform",
        e_dist=(0.0001, 0.98),
        i_dist=(0.0,  180.0),  i_mode="uniform",
        diameter_log_dist=(2.0, 1.5),
        albedo_dist=(15.0, 3.0),
        density_log_dist=(-7.5, 0.8),
        rotation_log_dist=(5.0, 1.5),
        a2_log_dist=(-6.5, 1.0),         # extreme A2
        a1_fraction_of_a2=1.5,
        signature_strength=0.999,
    ))

    scenarios.append(ScenarioSpec(
        tier=KardashevTier.K2_0,
        name="network_statistical_anomaly",
        description=(
            "Individual element that appears near-natural in isolation, but "
            "the *population* creates a network sigma >> 5.  Detected only "
            "via BC11 pattern analysis, not single-object sigma.  Designed "
            "to test whether the BC11 pipeline catches engineered clustering."
        ),
        a_dist=(1.5,  2.8),    a_mode="uniform",
        e_dist=(0.05,  0.25),
        # Extreme node clustering: tight normal around 0° or 180°
        i_dist=(2.0, 0.5),     i_mode="normal",   # i ≈ 2° ± 0.5
        diameter_log_dist=(-1.5, 0.5),
        albedo_dist=(4.0, 3.0),
        density_log_dist=(-1.5, 0.5),    # near-natural density (disguised)
        rotation_log_dist=(1.5, 0.7),
        a2_log_dist=(-14.5, 0.8),        # near-natural A2 (disguised)
        a1_fraction_of_a2=0.2,
        signature_strength=0.40,  # individual σ low; network σ >> 5
    ))

    return scenarios


# Singleton catalogue
SCENARIO_CATALOGUE: List[ScenarioSpec] = _build_scenario_catalogue()

# Index by (tier, name)
SCENARIO_INDEX: Dict[Tuple[KardashevTier, str], ScenarioSpec] = {
    (s.tier, s.name): s for s in SCENARIO_CATALOGUE
}


# ===========================================================================
# SyntheticNEO — output record
# ===========================================================================

@dataclass
class SyntheticNEO:
    """
    One synthetic artificial NEO sample, fully labelled.

    The `to_neo_data()` method converts this to an `aneos_core.data.models.NEOData`
    instance so it can flow through the existing `FeatureEngineer` pipeline.
    """
    designation: str
    tier: KardashevTier
    scenario: str
    scenario_description: str

    # Raw parameter values (orbital, physical, non-grav)
    params: Dict[str, Any]

    # Derived anomaly flags (informational)
    anomaly_flags: Dict[str, bool] = field(default_factory=dict)

    # Label always "artificial" — tier/scenario give finer granularity
    label: str = "artificial"
    label_source: str = "synthetic_kardashev"
    signature_strength: float = 0.8

    def to_neo_data(self):
        """
        Convert to an aneos_core.data.models.NEOData instance.

        Imports are deferred so this module is usable without the full
        aneos_core data package installed.
        """
        from aneos_core.data.models import (
            NEOData, OrbitalElements, PhysicalProperties,
            NonGravitationalParameters,
        )

        oe = OrbitalElements(
            designation=self.designation,
            semi_major_axis=self.params["a"],
            eccentricity=self.params["e"],
            inclination=self.params["i"],
            ra_of_ascending_node=self.params["om"],
            arg_of_periapsis=self.params["w"],
            mean_anomaly=self.params["M"],
            ascending_node=self.params["om"],
            argument_of_perihelion=self.params["w"],
        )

        pp = PhysicalProperties(
            diameter_km=self.params["diameter_km"],
            albedo=self.params["albedo"],
            density_g_cm3=self.params["density_g_cm3"],
            rotation_period_hours=self.params["rotation_period_hours"],
            absolute_magnitude_h=self.params["absolute_magnitude_h"],
        )

        neo = NEOData(
            designation=self.designation,
            orbital_elements=oe,
            physical_properties=pp,
            sources_used=["synthetic_kardashev"],
        )
        # Attach non-grav params as a metadata attribute for downstream use
        neo.nongrav_params = {
            "A1": self.params["nongrav_A1"],
            "A2": self.params["nongrav_A2"],
            "A3": self.params["nongrav_A3"],
        }
        return neo

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        Flat dict of all features, suitable for constructing a pandas DataFrame
        or a numpy feature vector without going through the full FeatureEngineer.
        """
        p = self.params
        d = {
            "designation":         self.designation,
            "tier":                self.tier.value,
            "scenario":            self.scenario,
            "label":               self.label,
            # Orbital
            "a":                   p["a"],
            "e":                   p["e"],
            "i":                   p["i"],
            "om":                  p["om"],
            "w":                   p["w"],
            "M":                   p["M"],
            "perihelion_q":        p["a"] * (1 - p["e"]),
            "aphelion_Q":          p["a"] * (1 + p["e"]),
            "orbital_period_yr":   p["a"] ** 1.5,
            # Physical
            "diameter_km":         p["diameter_km"],
            "albedo":              p["albedo"],
            "density_g_cm3":       p["density_g_cm3"],
            "rotation_period_h":   p["rotation_period_hours"],
            "abs_mag_H":           p["absolute_magnitude_h"],
            # Derived
            "area_to_mass_proxy":  p["diameter_km"] ** 2 / max(
                                       p["density_g_cm3"] * (p["diameter_km"] / 2) ** 3,
                                       1e-30),
            # Non-grav
            "nongrav_A1":          p["nongrav_A1"],
            "nongrav_A2":          p["nongrav_A2"],
            "nongrav_A3":          p["nongrav_A3"],
            "nongrav_magnitude":   (p["nongrav_A1"] ** 2 + p["nongrav_A2"] ** 2) ** 0.5,
            # Anomaly flags
            **{f"flag_{k}": int(v) for k, v in self.anomaly_flags.items()},
        }
        return d


# ===========================================================================
# Generator
# ===========================================================================

class KardashevSyntheticGenerator:
    """
    Generates balanced batches of synthetic artificial NEOs from all 14
    Kardashev scenarios for use as ML training positives.

    Usage
    -----
    >>> gen = KardashevSyntheticGenerator(seed=42)
    >>> samples = gen.generate(n_per_scenario=100)
    >>> len(samples)  # 14 scenarios × 100 = 1400
    1400
    >>> samples[0].tier
    <KardashevTier.K0_5: 'K0.5'>
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._scenarios = SCENARIO_CATALOGUE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_per_scenario: int = 100,
        tiers: Optional[List[KardashevTier]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> List[SyntheticNEO]:
        """
        Generate synthetic NEOs from all (or a subset of) scenarios.

        Parameters
        ----------
        n_per_scenario:
            Number of samples per scenario.
        tiers:
            If given, only generate from the specified tiers.
        scenarios:
            If given, only generate from the named sub-scenarios.

        Returns
        -------
        List[SyntheticNEO] sorted by tier then scenario name.
        """
        selected = self._scenarios
        if tiers is not None:
            selected = [s for s in selected if s.tier in tiers]
        if scenarios is not None:
            selected = [s for s in selected if s.name in scenarios]

        results: List[SyntheticNEO] = []
        for spec in selected:
            for idx in range(n_per_scenario):
                designation = f"SYNTH-{spec.tier.value}-{spec.name[:8].upper()}-{idx:04d}"
                params = spec.sample(self.rng)
                flags = self._compute_anomaly_flags(params)
                results.append(SyntheticNEO(
                    designation=designation,
                    tier=spec.tier,
                    scenario=spec.name,
                    scenario_description=spec.description,
                    params=params,
                    anomaly_flags=flags,
                    signature_strength=spec.signature_strength,
                ))

        logger.info(
            f"Generated {len(results)} synthetic NEOs "
            f"across {len(selected)} scenarios"
        )
        return results

    def generate_balanced(
        self,
        n_total: int = 1400,
    ) -> List[SyntheticNEO]:
        """
        Generate exactly n_total samples equally distributed across all scenarios.
        Rounds down to nearest multiple of scenario count.
        """
        n_per = max(1, n_total // len(self._scenarios))
        return self.generate(n_per_scenario=n_per)

    def scenario_summary(self) -> List[Dict[str, Any]]:
        """Return a human-readable summary of all defined scenarios."""
        return [
            {
                "tier":               s.tier.value,
                "name":               s.name,
                "description":        s.description,
                "signature_strength": s.signature_strength,
                "a_range_AU":         s.a_dist,
                "e_range":            s.e_dist,
                "i_range_deg":        s.i_dist,
                "a2_log10_range":     s.a2_log_dist,
            }
            for s in self._scenarios
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_anomaly_flags(params: Dict[str, Any]) -> Dict[str, bool]:
        """
        Flag which parameters lie outside the natural-population range.
        Used as additional features and for human interpretability.
        """
        a2_abs = abs(params["nongrav_A2"])
        return {
            "a_outside_neo_range":      not (_NATURAL_A_RANGE[0] <= params["a"] <= _NATURAL_A_RANGE[1]),
            "e_extreme":                params["e"] > _NATURAL_E_MAX,
            "i_retrograde":             params["i"] > 90.0,
            "albedo_extreme_high":      params["albedo"] > _NATURAL_ALBEDO_RANGE[1],
            "density_subnatural":       params["density_g_cm3"] < _NATURAL_DENSITY_RANGE[0],
            "a2_above_yarkovsky_max":   a2_abs > _NATURAL_A2_MAX,
            "circular_engineered":      params["e"] < 0.01,
        }


# ===========================================================================
# Training-dataset builder — ties generator to FeatureEngineer
# ===========================================================================

def build_labeled_feature_vectors(
    n_per_scenario: int = 50,
    seed: int = 42,
) -> Tuple[List, List[int], List[str]]:
    """
    Generate synthetic artificial NEOs, run them through FeatureEngineer,
    and return (feature_vectors, labels, designations).

    labels: 1 = artificial (synthetic), 0 = would be natural (not generated here)

    This function is intentionally simple — combine its output with a real
    natural-NEO feature set (from SBDB) before training:

        synt_fvs, synt_labels, _ = build_labeled_feature_vectors(50)
        # ... load real NEO feature vectors as natural_fvs, natural_labels = [0]*N
        X_train = FeatureEngineer().create_feature_matrix(synt_fvs + natural_fvs)
        y_train = np.array(synt_labels + natural_labels)

    Returns
    -------
    feature_vectors : List[FeatureVector]
    labels          : List[int]   (1 = artificial)
    designations    : List[str]
    """
    from aneos_core.ml.features import FeatureEngineer

    gen = KardashevSyntheticGenerator(seed=seed)
    synthetics = gen.generate(n_per_scenario=n_per_scenario)

    engineer = FeatureEngineer()
    feature_vectors = []
    labels = []
    designations = []

    for synth in synthetics:
        try:
            neo_data = synth.to_neo_data()
            fv = engineer.extract_features(neo_data)
            # Attach tier/scenario metadata for later analysis
            fv.metadata["label"] = 1
            fv.metadata["tier"] = synth.tier.value
            fv.metadata["scenario"] = synth.scenario
            fv.metadata["label_source"] = synth.label_source
            fv.metadata["signature_strength"] = synth.signature_strength
            feature_vectors.append(fv)
            labels.append(1)
            designations.append(synth.designation)
        except Exception as exc:
            logger.warning(f"Feature extraction failed for {synth.designation}: {exc}")

    logger.info(
        f"Built {len(feature_vectors)} labeled feature vectors "
        f"from {len(synthetics)} synthetic NEOs"
    )
    return feature_vectors, labels, designations


# ===========================================================================
# CLI / quick-look entry point
# ===========================================================================

def print_scenario_table() -> None:
    """Print a formatted table of all 14 scenarios to stdout."""
    gen = KardashevSyntheticGenerator()
    rows = gen.scenario_summary()
    header = f"{'Tier':<6}  {'Scenario':<28}  {'Sig.':>5}  {'a (AU)':<14}  {'e':<14}  A2 log10"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['tier']:<6}  {r['name']:<28}  {r['signature_strength']:>5.2f}  "
            f"{str(r['a_range_AU']):<14}  {str(r['e_range']):<14}  "
            f"{str(r['a2_log10_range'])}"
        )


if __name__ == "__main__":
    print_scenario_table()
    gen = KardashevSyntheticGenerator(seed=0)
    sample = gen.generate(n_per_scenario=1)
    for s in sample:
        print(f"\n{s.designation}  tier={s.tier.value}  scenario={s.scenario}")
        for k, v in s.to_feature_dict().items():
            if not k.startswith("flag_"):
                print(f"  {k:<28} = {v}")
        flags = {k: v for k, v in s.anomaly_flags.items() if v}
        if flags:
            print(f"  ANOMALY FLAGS: {list(flags.keys())}")
