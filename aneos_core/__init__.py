"""
aNEOS Core - Modular Near Earth Object Analysis System

This package provides modular research tooling for Near Earth Object evidence
and uncalibrated candidate screening. Scientific validity and origin require
independent evaluation; an application version is not a readiness certificate.

Key Components:
- config: Configuration management and settings
- data: Data fetching, caching, and source management
- analysis: Anomaly detection and statistical analysis
- reporting: Report generation and visualization
- utils: Common utilities and helper functions

Application identity: aneos_version.py (independent of API/model versions)
Original: neos_o3high_v6.19.1.py
"""

from aneos_version import APPLICATION_VERSION as __version__
__author__ = "aNEOS Development Team"

from .config.settings import ANEOSConfig, ConfigManager
from .data.models import NEOData, OrbitalElements
from .data.cache import CacheManager

__all__ = [
    'ANEOSConfig',
    'ConfigManager', 
    'NEOData',
    'OrbitalElements',
    'CacheManager'
]
