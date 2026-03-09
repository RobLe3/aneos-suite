"""
PA-6 RendezvousDetector — Stage 1 (REBOUND-free MOID pre-filter)

Detects statistically anomalous mutual close approach patterns between
Potentially Hazardous Asteroids (PHAs) without requiring orbit propagation.

Pipeline:
1. SBDB bulk query → all PHAs with orbital elements + A2 non-grav parameter
2. Drummond U_D pre-filter: cheap pair-distance proxy; discard pairs U_D > 0.2
3. Analytical period resonance: flag pairs with period ratio ≈ simple fraction
4. A2 correlation: flag pairs where both have significant A2 (same-sign anomaly)
5. Score and rank surviving pairs

This implements ADR-045 Stage 1. Stage 2 (REBOUND propagation) is deferred.
"""

import itertools
import math
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


@dataclass
class PHAObject:
    """Minimal orbital data for a Potentially Hazardous Asteroid."""
    pdes: str          # provisional designation
    a: float           # semi-major axis AU
    e: float           # eccentricity
    i: float           # inclination degrees
    om: float          # longitude of ascending node degrees
    w: float           # argument of perihelion degrees
    A2: Optional[float] = None  # non-gravitational A2 parameter


@dataclass
class RendezvousPair:
    """A flagged pair with anomalous mutual orbital proximity."""
    designation_a: str
    designation_b: str
    u_d: float           # Drummond distance (dimensionless)
    period_ratio: Optional[float] = None
    resonance: Optional[str] = None   # e.g. "1:1", "1:2"
    a2_correlated: bool = False
    anomaly_score: float = 0.0
    flags: List[str] = field(default_factory=list)


class PHAMoidScanner:
    """
    Scans all PHAs for anomalous pairwise orbital proximity.

    Usage:
        scanner = PHAMoidScanner()
        pairs = await scanner.run()
    """

    # Drummond U_D threshold — pairs beyond this are skipped
    UD_THRESHOLD = 0.12
    # Period resonance tolerance (fraction)
    RESONANCE_TOL = 0.03
    # Simple resonances to check: p:q where p,q in 1..4
    RESONANCES = [(p, q) for p in range(1, 5) for q in range(1, 5) if math.gcd(p, q) == 1]
    # A2 significance threshold (AU/day²) — values beyond this are anomalous
    A2_THRESHOLD = 1e-13

    def __init__(self):
        self.logger = logger

    async def fetch_phas(self) -> List[PHAObject]:
        """Query SBDB for all PHAs with orbital elements and A2."""
        if not _HAS_AIOHTTP:
            self.logger.error("aiohttp is not installed; cannot fetch PHAs from SBDB")
            return []

        params = {
            "sb-cdata": '{"AND":["Moid|LT|0.05","H|LT|22"]}',
            "fields": "pdes,a,e,i,om,w,A2",
            "limit": "3000",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    SBDB_QUERY_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as exc:
            self.logger.error(f"SBDB PHA query failed: {exc}")
            return []

        objects = []
        for row in data.get("data", []):
            try:
                pdes = str(row[0])
                a = float(row[1]) if row[1] else None
                e = float(row[2]) if row[2] else None
                inc = float(row[3]) if row[3] else None
                om = float(row[4]) if row[4] else None
                w = float(row[5]) if row[5] else None
                A2 = float(row[6]) if row[6] else None
                if None not in (a, e, inc, om, w):
                    objects.append(PHAObject(pdes=pdes, a=a, e=e, i=inc, om=om, w=w, A2=A2))
            except (ValueError, TypeError, IndexError):
                continue
        self.logger.info(f"Fetched {len(objects)} PHAs from SBDB")
        return objects

    @staticmethod
    def drummond_distance(obj_a: PHAObject, obj_b: PHAObject) -> float:
        """
        Drummond (1981) D-criterion — cheap proxy for orbital similarity.
        U_D = sqrt((e1-e2)² + (a1-a2)² + [2sin((i1-i2)/2)]² + ...)
        Pairs with U_D < UD_THRESHOLD are candidates for closer inspection.
        """
        da = obj_a.a - obj_b.a
        de = obj_a.e - obj_b.e
        di = math.radians(obj_a.i - obj_b.i)
        dom = math.radians(obj_a.om - obj_b.om)
        # simplified Drummond distance
        return math.sqrt(
            de**2
            + (da / max(obj_a.a, obj_b.a))**2
            + (2 * math.sin(di / 2))**2
            + (obj_a.e + obj_b.e)**2 * (2 * math.sin(dom / 2))**2 / 4
        )

    @staticmethod
    def period_au(a: float) -> float:
        """Orbital period in years from semi-major axis (Kepler's third law)."""
        return a ** 1.5

    def check_resonance(self, a1: float, a2: float) -> Optional[Tuple[str, float]]:
        """
        Return (resonance_string, ratio) if the period ratio is close to
        a simple fraction p:q (p,q ≤ 4), else None.
        """
        T1, T2 = self.period_au(a1), self.period_au(a2)
        ratio = T1 / T2 if T2 > 0 else 0.0
        for p, q in self.RESONANCES:
            target = p / q
            if abs(ratio - target) / target < self.RESONANCE_TOL:
                return (f"{p}:{q}", ratio)
        return None

    def score_pair(self, pair: RendezvousPair) -> float:
        """Combine signals into a single anomaly score [0,1]."""
        score = 0.0
        # Distance signal: closer U_D → higher score
        score += max(0.0, 1.0 - pair.u_d / self.UD_THRESHOLD) * 0.5
        if pair.resonance:
            score += 0.3
        if pair.a2_correlated:
            score += 0.2
        return min(score, 1.0)

    async def run(self, max_pairs: int = 50) -> List[RendezvousPair]:
        """
        Full scan: fetch PHAs → pairwise filter → resonance → A2 → score.
        Returns top max_pairs ranked by anomaly_score descending.
        """
        objects = await self.fetch_phas()
        if len(objects) < 2:
            return []

        candidates: List[RendezvousPair] = []
        n = len(objects)
        self.logger.info(
            f"Scanning {n*(n-1)//2:,} pairs with U_D threshold {self.UD_THRESHOLD}"
        )

        for obj_a, obj_b in itertools.combinations(objects, 2):
            try:
                ud = self.drummond_distance(obj_a, obj_b)
                if ud > self.UD_THRESHOLD:
                    continue

                flags: List[str] = ["proximity"]
                resonance_result = self.check_resonance(obj_a.a, obj_b.a)
                resonance_str = None
                ratio = None
                if resonance_result:
                    resonance_str, ratio = resonance_result
                    flags.append(f"resonance:{resonance_str}")

                a2_corr = (
                    obj_a.A2 is not None
                    and obj_b.A2 is not None
                    and abs(obj_a.A2) > self.A2_THRESHOLD
                    and abs(obj_b.A2) > self.A2_THRESHOLD
                    and obj_a.A2 * obj_b.A2 > 0  # same sign
                )
                if a2_corr:
                    flags.append("A2_correlated")

                pair = RendezvousPair(
                    designation_a=obj_a.pdes,
                    designation_b=obj_b.pdes,
                    u_d=ud,
                    period_ratio=ratio,
                    resonance=resonance_str,
                    a2_correlated=a2_corr,
                    flags=flags,
                )
                pair.anomaly_score = self.score_pair(pair)
                candidates.append(pair)
            except Exception:
                continue

        candidates.sort(key=lambda p: p.anomaly_score, reverse=True)
        self.logger.info(
            f"Found {len(candidates)} pairs below U_D threshold; returning top {max_pairs}"
        )
        return candidates[:max_pairs]
