"""BC11: frozen population evidence and shared screening (ADR-077).

Normal entrypoint: NetworkAnalysisSession. Legacy clustering/harmonic/correlation
and p-value fusion names remain exported for explicit historical diagnostics only.
They are not validated physical-origin or significance procedures.
"""

from .clustering import OrbitalElementClusterer, OrbitalCluster
from .harmonics import SynodicHarmonicAnalyzer, HarmonicSignal
from .correlation import NonGravCorrelator, CorrelationMatrix
from .network_sigma import NetworkSigmaCombiner
from .session import NetworkAnalysisSession, PatternAnalysisConfig, verify_network_report, network_report_to_markdown
from .rendezvous import PHAMoidScanner, RendezvousPair

__all__ = [
    "OrbitalElementClusterer", "OrbitalCluster",
    "SynodicHarmonicAnalyzer", "HarmonicSignal",
    "NonGravCorrelator", "CorrelationMatrix",
    "NetworkSigmaCombiner",
    "NetworkAnalysisSession", "PatternAnalysisConfig", "verify_network_report", "network_report_to_markdown",
    "PHAMoidScanner", "RendezvousPair",
]
