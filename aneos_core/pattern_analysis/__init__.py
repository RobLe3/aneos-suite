"""
Population Pattern Analysis — BC11.

Detects anomalies at the population level across a batch of NEOData objects.
Sub-modules: OrbitalElementClusterer (PA-1), SynodicHarmonicAnalyzer (PA-3),
NonGravCorrelator (PA-5), NetworkSigmaCombiner.
Pairwise RendezvousDetector (PA-6) is deferred — see ADR-045.
"""

from .clustering import OrbitalElementClusterer, OrbitalCluster
from .harmonics import SynodicHarmonicAnalyzer, HarmonicSignal
from .correlation import NonGravCorrelator, CorrelationMatrix
from .network_sigma import NetworkSigmaCombiner
from .session import NetworkAnalysisSession, PatternAnalysisConfig

__all__ = [
    "OrbitalElementClusterer", "OrbitalCluster",
    "SynodicHarmonicAnalyzer", "HarmonicSignal",
    "NonGravCorrelator", "CorrelationMatrix",
    "NetworkSigmaCombiner",
    "NetworkAnalysisSession", "PatternAnalysisConfig",
]
