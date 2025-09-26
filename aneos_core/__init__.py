"""
aNEOS Core - Modular Near Earth Object Analysis System

This package provides a modular, production-ready framework for analyzing
Near Earth Objects (NEOs) to identify potentially artificial characteristics.

Key Components:
- config: Configuration management and settings
- data: Data fetching, caching, and source management
- analysis: Anomaly detection and statistical analysis
- reporting: Report generation and visualization
- utils: Common utilities and helper functions

Version: 0.7.0 (Stabilization Series)
Original: neos_o3high_v6.19.1.py
"""

__version__ = "0.7.0"
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
