"""
Scientific constants and fixed values for aNEOS Core.

Contains astronomical constants, physical constants, and other
fixed values used throughout the NEO analysis system.
"""

import math

# Astronomical constants
AU = 149597870.7  # Astronomical Unit in kilometers
EARTH_RADIUS = 6371.0  # Earth radius in kilometers
EARTH_MASS = 5.972e24  # Earth mass in kilograms
SUN_MASS = 1.989e30  # Solar mass in kilograms
G = 6.67430e-11  # Gravitational constant in m³/kg/s²

# Orbital mechanics constants
SIDEREAL_DAY = 86164.0905  # Sidereal day in seconds
TROPICAL_YEAR = 365.24219  # Tropical year in days
OBLIQUITY = 23.43693  # Earth's obliquity in degrees

# NEO classification thresholds
NEO_MIN_PERIHELION = 1.3  # AU - minimum perihelion distance for NEO classification
PHA_MIN_DIAMETER = 140.0  # meters - minimum diameter for PHA classification
PHA_MAX_MOID = 0.05  # AU - maximum MOID for PHA classification

# Physical property ranges
TYPICAL_ASTEROID_DENSITY = 2.5  # g/cm³
MIN_ASTEROID_ALBEDO = 0.02
MAX_ASTEROID_ALBEDO = 0.9
ARTIFICIAL_ALBEDO_THRESHOLD = 0.6  # High albedo suggesting artificial origin

# Velocity thresholds (km/s)
EARTH_ESCAPE_VELOCITY = 11.2
SOLAR_ESCAPE_VELOCITY = 42.1
TYPICAL_NEO_VELOCITY = 20.0

# Geographic regions for analysis
GEOGRAPHIC_REGIONS = [
    {"name": "North America", "lat_range": (25, 75), "lon_range": (-170, -50)},
    {"name": "Europe", "lat_range": (35, 75), "lon_range": (-15, 45)},
    {"name": "Asia", "lat_range": (5, 75), "lon_range": (45, 180)},
    {"name": "Australia", "lat_range": (-45, -10), "lon_range": (110, 160)},
    {"name": "Africa", "lat_range": (-35, 40), "lon_range": (-20, 55)},
    {"name": "South America", "lat_range": (-60, 15), "lon_range": (-85, -30)},
]

# Data quality scoring weights
DATA_COMPLETENESS_WEIGHTS = {
    "orbital_elements": 0.3,
    "physical_properties": 0.2,
    "observation_history": 0.2,
    "close_approaches": 0.15,
    "discovery_info": 0.1,
    "spectral_data": 0.05,
}

# Statistical analysis parameters
STATISTICAL_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
OUTLIER_Z_SCORE_THRESHOLD = 2.5
CLUSTERING_MIN_SAMPLES = 3

# API rate limiting
DEFAULT_RATE_LIMIT = 10  # requests per second
BURST_RATE_LIMIT = 50  # requests per minute

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
ORBITAL_DATA_CACHE_TTL = 86400  # 24 hours in seconds
API_RESPONSE_CACHE_TTL = 1800  # 30 minutes in seconds

# Logging levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# File formats and extensions
SUPPORTED_DATA_FORMATS = [".json", ".csv", ".xml", ".txt"]
OUTPUT_FORMATS = ["json", "csv", "html", "pdf"]

# Anomaly detection categories
ANOMALY_CATEGORIES = [
    "orbital_mechanics",
    "velocity_shifts", 
    "close_approach_regularity",
    "purpose_driven",
    "physical_anomalies",
    "temporal_anomalies",
    "geographic_clustering",
    "acceleration_anomalies",
    "spectral_anomalies",
    "observation_history",
    "detection_history",
]

# Default scoring thresholds
ANOMALY_SCORE_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.9,
}

# Thread pool configuration
DEFAULT_MAX_WORKERS = 10
IO_BOUND_MAX_WORKERS = 20
CPU_BOUND_MAX_WORKERS = 4

# Retry configuration
DEFAULT_BACKOFF_FACTOR = 2.0
MAX_BACKOFF_DELAY = 300  # 5 minutes
DEFAULT_MAX_ATTEMPTS = 5