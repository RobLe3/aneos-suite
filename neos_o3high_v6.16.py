#!/usr/bin/env python3
# neos_analyzer.py
# Version: v6.16 (Beautified Output, Enhanced Logging & Reporting,
#                Fixed Source Verification & Horizons Polling,
#                Reduced Retries, Continuous Anomaly Scores)
# Date: 2025-02-16
#
# Overview:
# This script identifies potentially artificial NEOs through a multi‑stage process.
#
# Key enhancements in v6.16:
#  - Console output now uses dynamic text wrapping, colorization, emojis, and centered titles.
#  - File reports include decorative headers, section separators, and emoji-enhanced titles.
#  - Logging includes a clear session header and separator lines.
#  - When polling a source that fails, a log message is emitted indicating which source (SBDB, NEODyS, MPC, or Horizons) failed.
#  - Retry timer reduced to 3 seconds per retry with a maximum of 3 retries.
#  - Horizons polling now correctly extracts the inclination value (using key "i").
#  - Evaluation functions compute continuous scores (no binary clamping) and dynamic TAS is computed as the z‑score of raw TAS values.
#
# Workflow:
#   1. Source Verification (with enhanced logging)
#   2. CAD Data Fetching
#   3. Data Enrichment (poll all sources, with failure logging, and merge via completeness scoring)
#   4. Raw Anomaly Evaluation (continuous scoring)
#   5. Dynamic Statistical Analysis (using raw TAS z‑scores)
#   6. Reporting (console and file outputs with decorative formatting)

import os
import sys
import json
import logging
import re
import requests
import datetime
import time
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from skyfield.api import utc
from collections import defaultdict
import humanize  # pip install humanize
from astroquery.jplhorizons import Horizons  # pip install astroquery
from astroquery.mpc import MPC  # pip install astroquery
import threading
from astropy.coordinates import ICRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
from astropy.time import Time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import shelve
import traceback
import functools
import warnings
from erfa import ErfaWarning
from logging.handlers import RotatingFileHandler
import shutil, textwrap

# Suppress ERFA “dubious year” warnings.
warnings.filterwarnings("ignore", category=ErfaWarning)

# ==============================================================================
# HELPER FUNCTIONS FOR BEAUTIFIED OUTPUT
# ==============================================================================

def wrap_text(text: str, width: int = None) -> str:
    """Dynamically wrap text based on terminal width."""
    if width is None:
        width = shutil.get_terminal_size().columns - 4
    return textwrap.fill(text, width=width)

def colorize(text: str, color_code: str) -> str:
    """Wrap text with ANSI color codes. Example color codes: '36' (cyan), '35' (magenta),
    '32' (green), '31' (red)."""
    return f"\033[{color_code}m{text}\033[0m"

def print_wrapped(text: str, color: str = None):
    """Print wrapped text with optional ANSI color using tqdm.write."""
    wrapped = wrap_text(text)
    if color:
        wrapped = colorize(wrapped, color)
    tqdm.write(wrapped)

def log_separator():
    """Log a separator line."""
    logger.info("-" * 80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "DATA_NEOS_DIR": "dataneos",
    "DATA_DIR": os.path.join("dataneos", "data"),
    "ORBITAL_DIR": os.path.join("dataneos", "orbital_elements"),
    "LOG_FILE": os.path.join("dataneos", "neos_analyzer.log"),
    "OUTPUT_DIR": os.path.join("dataneos", "daily_outputs"),
    "CACHE_FILE": os.path.join("dataneos", "orbital_elements_cache"),
    "NEODYS_API_URL": "https://newton.spacedys.com/neodys/api/",
    "MPC_API_URL": "https://www.minorplanetcenter.net/",
    "HORIZONS_API_URL": "https://ssd.jpl.nasa.gov/api/horizons.api",
    "DATA_SOURCES_PRIORITY": ["SBDB", "NEODyS", "MPC", "Horizons"],
    "WEIGHTS": {
        "orbital_mechanics": 1.5,
        "velocity_shifts": 2.0,
        "close_approach_regularity": 2.0,
        "purpose_driven": 2.0,
        "physical_anomalies": 1.0,
        "temporal_anomalies": 1.0,
        "geographic_clustering": 1.0,
        "acceleration_anomalies": 2.0,
        "spectral_anomalies": 1.5,
        "observation_history": 1.0,
        "detection_history": 1.0
    },
    "THRESHOLDS": {
        "eccentricity": 0.8,
        "inclination": 45.0,
        "velocity_shift": 5.0,      # km/s
        "temporal_inertia": 100.0,
        "geo_eps": 5,
        "geo_min_samples": 2,
        "geo_min_clusters": 2,
        "diameter_min": 0.1,
        "diameter_max": 10.0,
        "albedo_min": 0.05,
        "albedo_max": 0.5,
        "min_subpoints": 2,
        "acceleration_threshold": 0.0005,  # km/s^2
        "observation_gap_multiplier": 3,
        "albedo_artificial": 0.6
    },
    "REQUEST_TIMEOUT": 10,
    "MAX_RETRIES": 3,         # Reduced from 5 to 3
    "INITIAL_RETRY_DELAY": 3  # Reduced from 5 to 3 seconds
}

# Global dictionaries to track polling stats and quality.
source_poll_stats = {source: {"success": 0, "failure": 0} for source in CONFIG["DATA_SOURCES_PRIORITY"]}
source_quality_stats = {source: 0.0 for source in CONFIG["DATA_SOURCES_PRIORITY"]}
source_quality_counts = {source: 0 for source in CONFIG["DATA_SOURCES_PRIORITY"]}

# Global cache for subpoints.
subpoint_cache = {}

# ==============================================================================
# SETUP & LOGGING FUNCTIONS
# ==============================================================================

def create_data_directories() -> None:
    for d in [CONFIG["DATA_NEOS_DIR"], CONFIG["DATA_DIR"], CONFIG["ORBITAL_DIR"], CONFIG["OUTPUT_DIR"]]:
        os.makedirs(d, exist_ok=True)

def setup_logging() -> logging.Logger:
    logger_obj = logging.getLogger("neos_analyzer")
    logger_obj.setLevel(logging.WARNING)
    handler = RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=1_000_000, backupCount=5)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_obj.addHandler(handler)
    # Write an initial log header.
    logger_obj.info("=" * 80)
    logger_obj.info("🚀 NEO Analyzer Log Session Started 🚀")
    logger_obj.info("=" * 80)
    return logger_obj

logger = setup_logging()
create_data_directories()

def load_api_key():
    return None

def parse_time_period(input_str: str):
    input_str = input_str.strip().lower()
    if input_str == "max":
        return relativedelta(years=200)
    pattern = r'^(\d+)([dmy])$'
    match = re.match(pattern, input_str)
    if not match:
        return None
    value, unit = match.groups()
    try:
        value = int(value)
    except ValueError:
        return None
    if unit == 'd':
        return relativedelta(days=value)
    elif unit == 'm':
        return relativedelta(months=value)
    elif unit == 'y':
        return relativedelta(years=value)
    return None

def define_geographic_regions() -> list:
    return [
        {'name': 'New York City, USA', 'lat': 40.7128, 'lon': -74.0060, 'radius_km': 50},
        {'name': 'Washington D.C., USA', 'lat': 38.9072, 'lon': -77.0369, 'radius_km': 50},
        {'name': 'Beijing, China', 'lat': 39.9042, 'lon': 116.4074, 'radius_km': 50},
        {'name': 'Cairo, Egypt', 'lat': 30.0444, 'lon': 31.2357, 'radius_km': 50},
        {'name': 'Tokyo, Japan', 'lat': 35.6895, 'lon': 139.6917, 'radius_km': 50},
        {'name': 'Sydney, Australia', 'lat': -33.8688, 'lon': 151.2093, 'radius_km': 50}
    ]

def create_session_with_retries() -> requests.Session:
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS"])
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

session = create_session_with_retries()

def verify_sources() -> dict:
    """
    Poll each source URL and log its availability.
    Treat 400-level responses as live.
    """
    available = {}
    for src in CONFIG["DATA_SOURCES_PRIORITY"]:
        if src == "NEODyS":
            url = CONFIG["NEODYS_API_URL"]
        elif src == "MPC":
            url = CONFIG["MPC_API_URL"]
        elif src == "Horizons":
            url = CONFIG["HORIZONS_API_URL"]
        else:
            url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
        try:
            response = session.get(url, timeout=CONFIG["REQUEST_TIMEOUT"])
            if response.ok or response.status_code == 400:
                available[src] = True
                logger.info(f"Source {src} is available (status: {response.status_code}).")
            else:
                available[src] = False
                logger.warning(f"Source {src} returned status code {response.status_code}. Marked as unavailable.")
        except Exception as e:
            available[src] = False
            logger.warning(f"Source check failed for {src}: {e}")
    return available

def is_passing_over_regions(subpoint, regions) -> int:
    if subpoint is None:
        return 0
    count = 0
    for region in regions:
        if geodesic(subpoint, (region["lat"], region["lon"])).kilometers <= region["radius_km"]:
            count += 1
    return count

def calculate_time_intervals(dates: list) -> list:
    intervals = []
    sorted_dates = sorted(dates)
    for i in range(1, len(sorted_dates)):
        delta = (sorted_dates[i] - sorted_dates[i-1]).days
        intervals.append(delta)
    return intervals

def safe_execute(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return None
    return wrapper

def wait_with_progress(delay, description="Waiting"):
    with tqdm(total=delay, desc=description, leave=False, bar_format='{desc}: {n_fmt}/{total_fmt} seconds') as pbar:
        for _ in range(delay):
            time.sleep(1)
            pbar.update(1)

# ==============================================================================
# CACHE CLEANUP FUNCTIONS
# ==============================================================================

def clear_orbital_cache():
    """Remove old cache files to avoid mixing data."""
    for root, dirs, files in os.walk(CONFIG["ORBITAL_DIR"]):
        for file in files:
            if file.endswith(".json"):
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file}: {e}")

def validate_cache_format(data: dict) -> bool:
    return isinstance(data, dict) and ("orbital_elements" in data or "epoch" in data)

def load_orbital_data_with_cleanup(designation: str, source: str):
    path = orbital_storage_path(designation, source)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not validate_cache_format(data):
                os.remove(path)
                return None
            return data
        except Exception as e:
            logger.warning(f"Cache file {path} error: {e}. Removing file.")
            os.remove(path)
            return None
    return None

def orbital_storage_path(designation: str, source: str) -> str:
    dir_path = os.path.join(CONFIG["ORBITAL_DIR"], designation)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{source}.json")

load_orbital_data = load_orbital_data_with_cleanup

# ==============================================================================
# SUBPOINT CALCULATION (Enhanced with Reduced Retries)
# ==============================================================================

def calculate_subpoint(orbital_elements: dict, close_approach_datetime: datetime.datetime, designation: str):
    cache_key = (designation, close_approach_datetime.isoformat())
    if cache_key in subpoint_cache:
        return subpoint_cache[cache_key]
    max_retries = CONFIG["MAX_RETRIES"]
    delay = CONFIG["INITIAL_RETRY_DELAY"]
    attempt = 0
    result = None
    while attempt < max_retries:
        try:
            obs_time = Time(close_approach_datetime)
            obj = Horizons(id=designation, location='@sun', epochs=obs_time.jd, id_type='smallbody')
            vectors = obj.vectors()
            if len(vectors) == 0:
                logger.warning(f"No vector data for {designation} at {close_approach_datetime}")
                result = None
                break
            x = vectors["x"][0] * u.au
            y = vectors["y"][0] * u.au
            z = vectors["z"][0] * u.au
            cart_rep = CartesianRepresentation(x=x, y=y, z=z)
            coord = ICRS(cart_rep)
            coord_earth = coord.transform_to(ITRS())
            subpoint = EarthLocation.from_geocentric(coord_earth.x, coord_earth.y, coord_earth.z, unit=u.au).geodetic
            result = (subpoint.lat.deg, subpoint.lon.deg)
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                attempt += 1
                logger.warning(f"503 error for {designation} at {close_approach_datetime} (attempt {attempt}/{max_retries}). Retrying in {delay} seconds.")
                wait_with_progress(delay, f"{designation} retry {attempt}/{max_retries}")
                delay = min(delay * 2, 3)
                continue
            else:
                logger.error(f"HTTP error for {designation} at {close_approach_datetime}: {e}")
                result = None
                break
        except ValueError as e:
            if "Ambiguous target name" in str(e):
                logger.error(f"Ambiguous target name for {designation} at {close_approach_datetime}: {e}")
                result = None
                break
            else:
                logger.error(f"ValueError for {designation} at {close_approach_datetime}: {e}")
                result = None
                break
        except Exception as e:
            if isinstance(e, (requests.exceptions.ConnectionError, ConnectionResetError)):
                attempt += 1
                logger.warning(f"Connection error for {designation} at {close_approach_datetime} (attempt {attempt}/{max_retries}). Retrying in {delay} seconds.")
                wait_with_progress(delay, f"{designation} retry {attempt}/{max_retries}")
                delay = min(delay * 2, 3)
                continue
            else:
                logger.error(f"Error calculating subpoint for {designation} at {close_approach_datetime}: {e}\n{traceback.format_exc()}")
                result = None
                break
    if result is None:
        logger.error(f"Failed to retrieve vectors for {designation} after {max_retries} attempts.")
    subpoint_cache[cache_key] = result
    return result

# ==============================================================================
# DATA FETCHING FUNCTIONS
# ==============================================================================

def fetch_cad_data(start_date: str, end_date: str, api_key=None):
    base_url = "https://ssd-api.jpl.nasa.gov/cad.api"
    params = {"date-min": start_date, "date-max": end_date, "sort": "date", "limit": 5000}
    try:
        response = session.get(base_url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
        response.raise_for_status()
        return response.json()
    except Exception as err:
        logger.error(f"Error fetching CAD data: {err}\n{traceback.format_exc()}")
    return None

@safe_execute
def save_orbital_data(designation: str, source: str, data: dict):
    path = orbital_storage_path(designation, source)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

@safe_execute
def fetch_orbital_elements_sbdb(designation: str, data_usage: dict) -> dict:
    base_url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    params = {"des": designation, "phys_par": 1}
    response = session.get(base_url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
    response.raise_for_status()
    data_usage["SBDB"] = data_usage.get("SBDB", 0) + len(response.content)
    data = response.json()
    if "orbit" in data:
        orbit = data["orbit"]
        orbital_data = {}
        for elem in orbit.get("elements", []):
            name = elem.get("name")
            value = elem.get("value")
            if name == "e":
                orbital_data["eccentricity"] = float(value) if value else None
            elif name == "i":
                orbital_data["inclination"] = float(value) if value else None
            elif name == "a":
                orbital_data["semi_major_axis"] = float(value) if value else None
            elif name == "node":
                orbital_data["ra_of_ascending_node"] = float(value) if value else None
            elif name == "w":
                orbital_data["arg_of_periapsis"] = float(value) if value else None
            elif name == "M":
                orbital_data["mean_anomaly"] = float(value) if value else None
            elif name == "epoch":
                try:
                    orbital_data["epoch"] = datetime.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=utc)
                except ValueError:
                    logger.warning(f"Invalid epoch format for {designation}: {value}")
                    orbital_data["epoch"] = None
        if "phys_par" in data:
            phys_par = data["phys_par"]
            orbital_data["diameter"] = float(phys_par.get("diameter", 0)) if phys_par.get("diameter") else None
            orbital_data["albedo"] = float(phys_par.get("albedo", 0)) if phys_par.get("albedo") else None
            orbital_data["rot_per"] = float(phys_par.get("rot_per", 0)) if phys_par.get("rot_per") else None
        return orbital_data
    else:
        logger.warning(f"No orbital data in SBDB for {designation}")
    return None

def key_map(key: str) -> str:
    mapping = {
        "e": "eccentricity",
        "i": "inclination",
        "a": "semi_major_axis",
        "node": "ra_of_ascending_node",
        "peri": "arg_of_periapsis",
        "M": "mean_anomaly",
        "epoch": "epoch"
    }
    return mapping.get(key, key)

@safe_execute
def fetch_orbital_elements_neodys(designation: str, data_usage: dict) -> dict:
    params = {"name": designation, "format": "json"}
    response = session.get(CONFIG["NEODYS_API_URL"], params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
    response.raise_for_status()
    data_usage["NEODyS"] = data_usage.get("NEODyS", 0) + len(response.content)
    data = response.json()
    if "orbit" in data:
        orbit = data["orbit"]
        orbital_data = {}
        for key in ["e", "i", "a", "node", "peri", "M", "epoch"]:
            value = orbit.get(key)
            if value:
                try:
                    if key == "epoch":
                        orbital_data["epoch"] = datetime.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=utc)
                    else:
                        orbital_data[key_map(key)] = float(value)
                except ValueError:
                    logger.warning(f"Invalid value for {key} in NEODyS for {designation}: {value}")
        return orbital_data
    else:
        logger.warning(f"No orbital data in NEODyS for {designation}")
    return None

@safe_execute
def fetch_orbital_elements_mpc(designation: str, data_usage: dict) -> dict:
    result = MPC.get_neos(designation=designation)
    if not result:
        logger.warning(f"No MPC data for {designation}")
        return None
    data = result[0]
    orbital_data = {
        "eccentricity": float(data["e"]),
        "inclination": float(data["incl"]),
        "semi_major_axis": float(data["a"]),
        "ra_of_ascending_node": float(data["Omega"]),
        "arg_of_periapsis": float(data["w"]),
        "mean_anomaly": float(data["M"]),
        "epoch": Time(data["epoch"], format="mjd").to_datetime().replace(tzinfo=utc),
        "diameter": float(data.get("diameter", 0)) if data.get("diameter") else None,
        "albedo": float(data.get("albedo", 0)) if data.get("albedo") else None,
        "rot_per": float(data.get("rot_per", 0)) if data.get("rot_per") else None
    }
    data_usage["MPC"] = data_usage.get("MPC", 0) + len(json.dumps(orbital_data))
    return orbital_data

@safe_execute
def fetch_orbital_elements_horizons(designation: str, data_usage: dict) -> dict:
    # Request heliocentric orbital elements from Horizons.
    obj = Horizons(id=designation, location='@sun', epochs='now')
    elements = obj.elements()
    if len(elements) == 0:
        logger.warning(f"No Horizons data for {designation}")
        return None
    el = elements[0]
    # Note: astroquery Horizons returns inclination under key "i"
    orbital_data = {
        "eccentricity": float(el["e"]),
        "inclination": float(el["i"]),
        "semi_major_axis": float(el["a"]),
        "ra_of_ascending_node": float(el["node"]),
        "arg_of_periapsis": float(el["peri"]),
        "mean_anomaly": float(el["M"]),
        "epoch": el.get("datetime", el.get("datetime_str")).replace(tzinfo=utc),
        "diameter": float(el.get("diameter", 0)) if el.get("diameter") else None,
        "albedo": float(el.get("albedo", 0)) if el.get("albedo") else None,
        "rot_per": float(el.get("rot_per", 0)) if el.get("rot_per") else None
    }
    data_usage["Horizons"] = data_usage.get("Horizons", 0) + len(json.dumps(orbital_data))
    return orbital_data

# ==============================================================================
# ENRICHMENT: POLL ALL SOURCES, INDEX, AND MERGE
# ==============================================================================

def compute_completeness(response: dict) -> float:
    required_keys = ["eccentricity", "inclination", "semi_major_axis",
                     "ra_of_ascending_node", "arg_of_periapsis",
                     "mean_anomaly", "epoch", "diameter", "albedo"]
    count = sum(1 for key in required_keys if key in response and response[key] is not None)
    return count / len(required_keys)

def merge_orbital_data(data_dict: dict) -> dict:
    required_keys = ["eccentricity", "inclination", "semi_major_axis",
                     "ra_of_ascending_node", "arg_of_periapsis",
                     "mean_anomaly", "epoch", "diameter", "albedo"]
    merged = {}
    completeness_scores = {src: compute_completeness(resp) for src, resp in data_dict.items() if resp}
    for key in required_keys:
        best = None
        best_score = -1
        for src, resp in data_dict.items():
            if resp and resp.get(key) is not None:
                score = completeness_scores.get(src, 0)
                if score > best_score:
                    best_score = score
                    best = resp.get(key)
        merged[key] = best
    return merged

def fetch_all_orbital_elements(designation: str, data_usage: dict, available_sources: dict) -> dict:
    results = {}
    for source in CONFIG["DATA_SOURCES_PRIORITY"]:
        if not available_sources.get(source, False):
            continue
        local = load_orbital_data(designation, source)
        if local is not None:
            results[source] = local
            source_poll_stats[source]['success'] += 1
            quality = compute_completeness(local)
            source_quality_stats[source] += quality
            source_quality_counts[source] += 1
        else:
            fetch_func = None
            if source == "SBDB":
                fetch_func = fetch_orbital_elements_sbdb
            elif source == "NEODyS":
                fetch_func = fetch_orbital_elements_neodys
            elif source == "MPC":
                fetch_func = fetch_orbital_elements_mpc
            elif source == "Horizons":
                fetch_func = fetch_orbital_elements_horizons
            if fetch_func:
                data = fetch_func(designation, data_usage)
                if data:
                    results[source] = data
                    save_orbital_data(designation, source, data)
                    source_poll_stats[source]['success'] += 1
                    quality = compute_completeness(data)
                    source_quality_stats[source] += quality
                    source_quality_counts[source] += 1
                else:
                    logger.warning(f"Failed to poll {source} for designation {designation}")
                    source_poll_stats[source]['failure'] += 1
    return results

def fetch_and_merge_orbital_elements(designation: str, cache, data_usage: dict, available_sources: dict) -> dict:
    if designation in cache:
        return cache[designation]
    responses = fetch_all_orbital_elements(designation, data_usage, available_sources)
    merged = merge_orbital_data(responses)
    result = {"orbital_elements": merged, 
              "sources_used": list(responses.keys()),
              "completeness": compute_completeness(merged)}
    source_contributions = {}
    for src in responses:
        if responses[src]:
            source_contributions[src] = compute_completeness(responses[src])
    result["source_contributions"] = source_contributions
    cache[designation] = result
    return result

# ==============================================================================
# RAW INDICATOR EVALUATION FUNCTIONS (Continuous Scoring)
# ==============================================================================

def evaluate_orbital_mechanics(designation: str, orbital_elements: dict, weights: dict, thresholds: dict) -> float:
    score = 0.0
    e = orbital_elements.get("eccentricity")
    i = orbital_elements.get("inclination")
    if e is not None:
        if e > thresholds["eccentricity"]:
            excess = (e - thresholds["eccentricity"]) / (1 - thresholds["eccentricity"])
            score += weights["orbital_mechanics"] * excess
    if i is not None:
        if i > thresholds["inclination"]:
            excess = (i - thresholds["inclination"]) / (90 - thresholds["inclination"])
            score += weights["orbital_mechanics"] * excess
    return score

def evaluate_velocity_shifts(cad_records: list, weights: dict, thresholds: dict) -> float:
    velocities_rel = []
    velocities_inf = []
    for record in cad_records:
        try:
            v_rel = float(record.get("v_rel") or 0)
            velocities_rel.append(v_rel)
        except (ValueError, TypeError):
            logger.warning("Invalid v_rel encountered.")
        try:
            v_inf = float(record.get("v_inf") or 0)
            velocities_inf.append(v_inf)
        except (ValueError, TypeError):
            logger.warning("Invalid v_inf encountered.")
    if len(velocities_rel) > 1:
        v_shift_rel = max(velocities_rel) - min(velocities_rel)
    else:
        v_shift_rel = 0.0
    if len(velocities_inf) > 1:
        v_shift_inf = max(velocities_inf) - min(velocities_inf)
    else:
        v_shift_inf = 0.0
    v_shift = max(v_shift_rel, v_shift_inf)
    if v_shift > thresholds["velocity_shift"]:
        excess = min((v_shift - thresholds["velocity_shift"]) / thresholds["velocity_shift"], 1.0)
        return weights["velocity_shifts"] * excess
    return 0.0

def evaluate_close_approach_regularity(cad_records: list, weights: dict) -> float:
    if len(cad_records) < 2:
        return 0.0
    intervals = calculate_time_intervals([r["cd"] for r in cad_records])
    if not intervals:
        return 0.0
    mean_int = np.mean(intervals)
    std_int = np.std(intervals)
    regularity = max(0, 1 - (std_int / mean_int))
    return weights["close_approach_regularity"] * regularity

def evaluate_purpose_driven_monitoring(cad_records: list, regions: list, weights: dict) -> float:
    score = 0.0
    for record in cad_records:
        regions_over = is_passing_over_regions(record.get("subpoint"), regions)
        score += weights["purpose_driven"] * (regions_over / len(regions))
    return score

def evaluate_physical_anomalies(designation: str, orbital_elements: dict, weights: dict, thresholds: dict) -> float:
    score = 0.0
    diameter = orbital_elements.get("diameter")
    albedo = orbital_elements.get("albedo")
    if diameter is not None:
        if diameter < thresholds["diameter_min"]:
            diff = (thresholds["diameter_min"] - diameter) / thresholds["diameter_min"]
            score += weights["physical_anomalies"] * diff
        elif diameter > thresholds["diameter_max"]:
            diff = (diameter - thresholds["diameter_max"]) / thresholds["diameter_max"]
            score += weights["physical_anomalies"] * diff
    if albedo is not None:
        if albedo < thresholds["albedo_min"]:
            diff = (thresholds["albedo_min"] - albedo) / thresholds["albedo_min"]
            score += weights["physical_anomalies"] * diff
        elif albedo > thresholds["albedo_max"]:
            diff = (albedo - thresholds["albedo_max"]) / thresholds["albedo_max"]
            score += weights["physical_anomalies"] * diff
    return score

def evaluate_temporal_anomalies(cad_records: list, weights: dict, thresholds: dict) -> float:
    if len(cad_records) < 2:
        return 0.0
    intervals = calculate_time_intervals([r["cd"] for r in cad_records])
    if not intervals:
        return 0.0
    inertia = np.std(intervals)
    if inertia < thresholds["temporal_inertia"]:
        return weights["temporal_anomalies"] * (1 - inertia/thresholds["temporal_inertia"])
    return 0.0

def evaluate_geographic_clustering(cad_records: list, weights: dict, thresholds: dict) -> float:
    subpoints = [r["subpoint"] for r in cad_records if r.get("subpoint")]
    if len(subpoints) < thresholds["min_subpoints"]:
        return 0.0
    coords = np.array(subpoints)
    clustering = DBSCAN(eps=thresholds["geo_eps"], min_samples=thresholds["geo_min_samples"]).fit(coords)
    clusters = [label for label in clustering.labels_ if label != -1]
    if not clusters:
        return 0.0
    ratio = len(set(clusters)) / len(subpoints)
    return weights["geographic_clustering"] * (1 - ratio)

def evaluate_acceleration_anomalies(cad_records: list, weights: dict, thresholds: dict) -> float:
    if len(cad_records) < 2:
         return 0.0
    accelerations = []
    for i in range(1, len(cad_records)):
         try:
             v1 = float(cad_records[i-1].get("v_rel") or 0)
             v2 = float(cad_records[i].get("v_rel") or 0)
         except (ValueError, TypeError):
             continue
         t1 = cad_records[i-1].get("cd")
         t2 = cad_records[i].get("cd")
         delta_t = (t2 - t1).total_seconds()
         if delta_t <= 0:
             continue
         accel = abs(v2 - v1) / delta_t
         accelerations.append(accel)
    if accelerations:
         max_accel = max(accelerations)
         if max_accel > thresholds["acceleration_threshold"]:
             excess = min((max_accel - thresholds["acceleration_threshold"]) / thresholds["acceleration_threshold"], 1.0)
             return weights.get("acceleration_anomalies", 0) * excess
    return 0.0

def evaluate_spectral_anomalies(orbital_elements: dict, weights: dict, thresholds: dict) -> float:
    score = 0.0
    spectral_type = orbital_elements.get("spectral_type")
    albedo = orbital_elements.get("albedo")
    if spectral_type is None and albedo is None:
         return score
    if spectral_type is None and albedo is not None:
         if albedo > 0.4:
              spectral_type = "metallic"
         elif albedo < 0.1:
              spectral_type = "carbonaceous"
         else:
              spectral_type = "stony"
         orbital_elements["spectral_type"] = spectral_type
    if spectral_type == "metallic" and albedo is not None and albedo > thresholds.get("albedo_artificial", 0.6):
         diff = (albedo - thresholds.get("albedo_artificial", 0.6)) / (1 - thresholds.get("albedo_artificial", 0.6))
         score += weights.get("spectral_anomalies", 0) * diff
    return score

def evaluate_observation_gaps(cad_records: list, weights: dict, thresholds: dict) -> float:
    if len(cad_records) < 2:
         return 0.0
    intervals = calculate_time_intervals([r["cd"] for r in cad_records])
    if not intervals:
         return 0.0
    median_gap = np.median(intervals)
    max_gap = max(intervals)
    if max_gap > thresholds.get("observation_gap_multiplier", 3) * median_gap:
         excess = (max_gap - thresholds.get("observation_gap_multiplier", 3) * median_gap) / max_gap
         return weights.get("observation_history", 0) * excess
    return 0.0

def evaluate_detection_history(sources_used: list, weights: dict) -> float:
    ratio = len(sources_used) / len(CONFIG["DATA_SOURCES_PRIORITY"])
    return weights.get("detection_history", 0) * ratio

def evaluate_anomaly_score(neo: dict, cad_records: list, regions: list, weights: dict, thresholds: dict) -> (float, dict):
    orbital_elements = neo["orbital_elements"]
    sources_used = neo.get("sources_used", [])
    score_components = {
        "orbital_mechanics": evaluate_orbital_mechanics(neo["designation"], orbital_elements, weights, thresholds),
        "velocity_shifts": evaluate_velocity_shifts(cad_records, weights, thresholds),
        "close_approach_regularity": evaluate_close_approach_regularity(cad_records, weights),
        "purpose_driven": evaluate_purpose_driven_monitoring(cad_records, regions, weights),
        "physical_anomalies": evaluate_physical_anomalies(neo["designation"], orbital_elements, weights, thresholds),
        "temporal_anomalies": evaluate_temporal_anomalies(cad_records, weights, thresholds),
        "geographic_clustering": evaluate_geographic_clustering(cad_records, weights, thresholds),
        "acceleration_anomalies": evaluate_acceleration_anomalies(cad_records, weights, thresholds),
        "spectral_anomalies": evaluate_spectral_anomalies(orbital_elements, weights, thresholds),
        "observation_history": evaluate_observation_gaps(cad_records, weights, thresholds),
        "detection_history": evaluate_detection_history(sources_used, weights)
    }
    raw_TAS = sum(score_components.values())
    return raw_TAS, score_components

# ==============================================================================
# DYNAMIC STATISTICAL ANALYSIS
# ==============================================================================

def compute_dynamic_TAS(enriched_neos: list) -> (dict, dict):
    raw_TAS_values = [neo["raw_TAS"] for neo in enriched_neos]
    mean_val = np.mean(raw_TAS_values) if raw_TAS_values else 0
    std_val = np.std(raw_TAS_values)
    std_val = std_val if std_val != 0 else 1
    category_counts = defaultdict(int)
    for neo in enriched_neos:
        dynamic_TAS = (neo["raw_TAS"] - mean_val) / std_val
        neo["dynamic_TAS"] = dynamic_TAS
        if dynamic_TAS < -1:
            category = "Within Normal Range"
        elif dynamic_TAS < 0:
            category = "Slightly Anomalous"
        elif dynamic_TAS < 1:
            category = "Moderately Anomalous"
        elif dynamic_TAS < 2:
            category = "Highly Anomalous"
        else:
            category = "Extremely Anomalous / Potentially Artificial"
        neo["dynamic_category"] = category
        category_counts[category] += 1
    stats = {"mean": mean_val, "std": std_val}
    return category_counts, stats

def dynamic_analysis(enriched_neos: list) -> (dict, dict, dict):
    dynamic_category_counts, indicator_stats = compute_dynamic_TAS(enriched_neos)
    dynamic_TAS_values = [neo["dynamic_TAS"] for neo in enriched_neos]
    highest_dynamic_TAS = max(dynamic_TAS_values) if dynamic_TAS_values else None
    lowest_dynamic_TAS = min(dynamic_TAS_values) if dynamic_TAS_values else None
    top10 = sorted(enriched_neos, key=lambda x: x.get("dynamic_TAS", 0), reverse=True)[:10]
    dynamic_stats = {
        "Highest Dynamic TAS": highest_dynamic_TAS,
        "Lowest Dynamic TAS": lowest_dynamic_TAS,
        "Top 10 High Dynamic TAS NEOs": top10
    }
    return dynamic_category_counts, indicator_stats, dynamic_stats

# ==============================================================================
# REPORTING FUNCTIONS
# ==============================================================================

def print_beautified_console_summary(enriched_neos: list, raw_stats: dict, dynamic_category_counts: dict, dynamic_stats: dict) -> None:
    border = "=" * 60
    header = colorize("🚀 FINAL ANALYSIS SUMMARY (Anomalous NEOs Only) 🚀".center(60), "35")
    print("\n" + border)
    print(header)
    print(border)
    print(f"Total NEOs Analyzed: {raw_stats['Total NEOs Analyzed']}")
    print(f"Average Raw TAS: {raw_stats['Average Raw TAS']:.2f}")
    print(f"Highest Raw TAS: {raw_stats['Highest Raw TAS']}")
    print(f"Lowest Raw TAS: {raw_stats['Lowest Raw TAS']}")
    print("\nDynamic Analysis:")
    print(f"Highest Dynamic TAS: {dynamic_stats['Highest Dynamic TAS']}")
    print(f"Lowest Dynamic TAS: {dynamic_stats['Lowest Dynamic TAS']}")
    print("\nDynamic Category Counts:")
    for cat, cnt in dynamic_category_counts.items():
        print(f"  {cat}: {cnt}")
    print("\nTop 10 High Dynamic TAS NEOs:")
    for neo in dynamic_stats["Top 10 High Dynamic TAS NEOs"]:
        print(f"  {neo['designation']} - Dynamic TAS: {neo.get('dynamic_TAS', 0):.2f} - Category: {neo.get('dynamic_category', 'N/A')}")
    print("\nSource Polling Statistics:")
    total_quality = sum(source_quality_stats.values())
    for src in CONFIG["DATA_SOURCES_PRIORITY"]:
        succ = source_poll_stats[src]['success']
        fail = source_poll_stats[src]['failure']
        count = source_quality_counts[src]
        avg_quality = (source_quality_stats[src] / count) if count > 0 else 0
        perc = (source_quality_stats[src] / total_quality * 100) if total_quality > 0 else 0
        print(f"  {src}: Successes = {succ}, Failures = {fail}, Avg Quality = {avg_quality:.2f}, Contribution = {perc:.1f}%")
    print(border + "\n")

def save_results(enriched_neos: list, raw_stats: dict, dynamic_category_counts: dict, dynamic_stats: dict, data_usage: dict) -> None:
    base_output = os.path.join(CONFIG["OUTPUT_DIR"],
                               f"scan_{datetime.date.today().isoformat()}_{datetime.datetime.now().strftime('%H%M%S')}")
    os.makedirs(base_output, exist_ok=True)
    
    # Write a decorative header with emojis for the summary report.
    summary_header = "🚀 FINAL ANALYSIS REPORT 🚀".center(60)
    summary_filepath = os.path.join(base_output, "analysis_summary.txt")
    try:
        with open(summary_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(summary_header + "\n")
            f.write("=" * 60 + "\n\n")
            f.write("Raw Analysis Statistics:\n")
            for key, value in raw_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nDynamic Category Counts:\n")
            for cat, cnt in dynamic_category_counts.items():
                f.write(f"  {cat}: {cnt}\n")
            f.write("\nDynamic Analysis:\n")
            f.write(f"  Highest Dynamic TAS: {dynamic_stats['Highest Dynamic TAS']}\n")
            f.write(f"  Lowest Dynamic TAS: {dynamic_stats['Lowest Dynamic TAS']}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing analysis summary: {e}\n{traceback.format_exc()}")
    
    details_filepath = os.path.join(base_output, "detailed_results.txt")
    try:
        with open(details_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("📝 DETAILED RESULTS 📝".center(60) + "\n")
            f.write("=" * 60 + "\n\n")
            for neo in enriched_neos:
                f.write(f"Designation: {neo['designation']}\n")
                f.write(f"  Raw TAS: {neo.get('raw_TAS', 'N/A')}\n")
                f.write(f"  Dynamic TAS: {neo.get('dynamic_TAS', 'N/A'):.2f}\n")
                f.write(f"  Dynamic Category: {neo.get('dynamic_category', 'N/A')}\n")
                f.write(f"  Close Approaches: {len(neo.get('close_approaches', []))}\n")
                f.write(f"  Observation Period: {neo.get('first_observation')} to {neo.get('last_observation')}\n")
                f.write(f"  Score Components: {neo.get('score_components', {})}\n")
                f.write("-" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing detailed results: {e}\n{traceback.format_exc()}")
    
    top10_filepath = os.path.join(base_output, "top10_high_dynamic_TAS.txt")
    try:
        with open(top10_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("📊 TOP 10 HIGH DYNAMIC TAS NEOs 📊".center(60) + "\n")
            f.write("=" * 60 + "\n\n")
            for neo in dynamic_stats["Top 10 High Dynamic TAS NEOs"]:
                f.write(f"Designation: {neo['designation']}\n")
                f.write(f"  Dynamic TAS: {neo.get('dynamic_TAS', 'N/A'):.2f}\n")
                f.write(f"  Dynamic Category: {neo.get('dynamic_category', 'N/A')}\n")
                f.write(f"  Raw TAS: {neo.get('raw_TAS', 'N/A')}\n")
                f.write("-" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing top 10 results: {e}\n{traceback.format_exc()}")
    
    usage_filepath = os.path.join(base_output, "data_usage_report.txt")
    try:
        with open(usage_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("📊 DATA USAGE REPORT 📊".center(60) + "\n")
            f.write("=" * 60 + "\n\n")
            for source, bytes_used in data_usage.items():
                f.write(f"  {source}: {humanize.naturalsize(bytes_used, binary=True)}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing data usage report: {e}\n{traceback.format_exc()}")
    
    source_stats_filepath = os.path.join(base_output, "source_poll_stats.txt")
    try:
        with open(source_stats_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("📊 SOURCE POLLING STATISTICS 📊".center(60) + "\n")
            f.write("=" * 60 + "\n\n")
            total_quality = sum(source_quality_stats.values())
            for src in CONFIG["DATA_SOURCES_PRIORITY"]:
                succ = source_poll_stats[src]['success']
                fail = source_poll_stats[src]['failure']
                count = source_quality_counts[src]
                avg_quality = (source_quality_stats[src] / count) if count > 0 else 0
                perc = (source_quality_stats[src] / total_quality * 100) if total_quality > 0 else 0
                f.write(f"{src}: Successes = {succ}, Failures = {fail}, Avg Quality = {avg_quality:.2f}, Contribution = {perc:.1f}%\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing source polling stats: {e}\n{traceback.format_exc()}")
    
    print_wrapped(f"🚀 Results saved to {base_output}", "32")

# ==============================================================================
# FILE I/O FUNCTIONS (Parallel JSON Loading)
# ==============================================================================

def load_single_json_file(filepath: str) -> list:
    records = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if "data" in data:
            for cad_record in data["data"]:
                if len(cad_record) < 9:
                    logger.warning(f"Insufficient CAD record: {cad_record}")
                    continue
                designation = cad_record[0].strip()
                if not designation:
                    continue
                try:
                    cad_datetime = datetime.datetime.strptime(cad_record[3], "%Y-%b-%d %H:%M").replace(tzinfo=utc)
                except ValueError:
                    logger.warning(f"Invalid datetime in CAD record: {cad_record[3]}")
                    continue
                records.append({
                    "designation": designation,
                    "orbit_id": cad_record[1],
                    "cd": cad_datetime,
                    "dist": cad_record[4],
                    "v_rel": cad_record[7],
                    "v_inf": cad_record[8]
                })
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}\n{traceback.format_exc()}")
    return records

def load_json_files(start_date: str, end_date: str) -> list:
    neos = []
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    file_paths = []
    for year in os.listdir(CONFIG["DATA_DIR"]):
        year_path = os.path.join(CONFIG["DATA_DIR"], year)
        if os.path.isdir(year_path):
            for filename in os.listdir(year_path):
                if filename.endswith(".json"):
                    try:
                        file_year, file_month = map(int, filename.rstrip(".json").split("-"))
                        file_date = datetime.date(file_year, file_month, 1)
                        if file_date < start_dt.replace(day=1) or file_date > end_dt.replace(day=1):
                            continue
                        file_paths.append(os.path.join(year_path, filename))
                    except ValueError:
                        logger.warning(f"Filename {filename} invalid. Skipping.")
                        continue
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_single_json_file, fp) for fp in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading JSON Files", unit="file"):
            neos.extend(future.result())
    return neos

# ==============================================================================
# DATA ENRICHMENT & RAW SCORE EVALUATION
# ==============================================================================

def compute_subpoint_for_record(neo: dict, record: dict, available_sources: dict) -> dict:
    if not available_sources.get("Horizons", False):
        record["subpoint"] = None
    else:
        record["subpoint"] = calculate_subpoint(neo["orbital_elements"], record["cd"], neo["designation"])
    try:
        record["distance_km"] = float(record["dist"]) * 149597870.7
    except Exception:
        record["distance_km"] = None
    try:
        record["relative_velocity_km_s"] = float(record.get("v_rel") or 0)
    except Exception:
        record["relative_velocity_km_s"] = None
    try:
        record["infinity_velocity_km_s"] = float(record.get("v_inf") or 0)
    except Exception:
        record["infinity_velocity_km_s"] = None
    return record

def analyze_data(regions: list, tas_threshold: float, start_date: str, end_date: str, data_usage: dict, available_sources: dict) -> (list, dict):
    neos = load_json_files(start_date, end_date)
    neo_map = {}
    for rec in neos:
        desig = rec["designation"]
        if desig not in neo_map:
            neo_map[desig] = {"close_approaches": [], "first_observation": rec["cd"], "last_observation": rec["cd"]}
        neo_map[desig]["close_approaches"].append(rec)
        if rec["cd"] < neo_map[desig]["first_observation"]:
            neo_map[desig]["first_observation"] = rec["cd"]
        if rec["cd"] > neo_map[desig]["last_observation"]:
            neo_map[desig]["last_observation"] = rec["cd"]
    
    enriched_neos = []
    weights = CONFIG["WEIGHTS"]
    thresholds = CONFIG["THRESHOLDS"]
    max_workers = 10
    cache = shelve.open(CONFIG["CACHE_FILE"])
    with tqdm(total=len(neo_map), desc="Enriching NEOs", position=1, unit="neo") as pbar_enrich:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_and_merge_orbital_elements, desig, cache, data_usage, available_sources): desig
                       for desig in neo_map.keys()}
            for future in as_completed(futures):
                desig = futures[future]
                try:
                    enrich_data = future.result()
                except Exception as e:
                    logger.error(f"Error enriching {desig}: {e}\n{traceback.format_exc()}")
                    pbar_enrich.update(1)
                    continue
                if not enrich_data:
                    pbar_enrich.update(1)
                    continue
                cad_records = neo_map[desig]["close_approaches"]
                enriched_neos.append({
                    "designation": desig,
                    "orbital_elements": enrich_data["orbital_elements"],
                    "sources_used": enrich_data["sources_used"],
                    "source_contributions": enrich_data.get("source_contributions", {}),
                    "close_approaches": cad_records,
                    "first_observation": neo_map[desig]["first_observation"],
                    "last_observation": neo_map[desig]["last_observation"]
                })
                pbar_enrich.update(1)
    cache.close()
    
    subpoint_tasks = []
    with ThreadPoolExecutor(max_workers=20) as sub_executor:
        for neo in enriched_neos:
            for rec in neo["close_approaches"]:
                future = sub_executor.submit(compute_subpoint_for_record, neo, rec, available_sources)
                subpoint_tasks.append(future)
        with tqdm(total=len(subpoint_tasks), desc="Polling Subpoints", unit="subpoint", position=2) as polling_pbar:
            for _ in as_completed(subpoint_tasks):
                polling_pbar.update(1)
    
    final_enriched = []
    for neo in enriched_neos:
        cad_records = neo["close_approaches"]
        raw_TAS, comp_scores = evaluate_anomaly_score(neo, cad_records, regions, weights, thresholds)
        neo["raw_TAS"] = raw_TAS
        neo["score_components"] = comp_scores
        final_enriched.append(neo)
    
    raw_stats = {
        "Total NEOs Analyzed": len(final_enriched),
        "Average Raw TAS": np.mean([neo["raw_TAS"] for neo in final_enriched]) if final_enriched else 0,
        "Highest Raw TAS": max([neo["raw_TAS"] for neo in final_enriched]) if final_enriched else 0,
        "Lowest Raw TAS": min([neo["raw_TAS"] for neo in final_enriched]) if final_enriched else 0
    }
    return final_enriched, raw_stats

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================

def main():
    try:
        clear_orbital_cache()
        api_key = load_api_key()
        available_sources = verify_sources()
        # Console welcome message with emoji and dynamic wrapping.
        print_wrapped("🚀 Welcome to the NEO Analyzer! 🚀", "36")
        while True:
            tp_input = input("Enter time period for analysis (e.g., '1d', '1m', '1y', '200y' or 'max'): ").strip().lower()
            delta = parse_time_period(tp_input)
            if delta:
                break
            else:
                print_wrapped("❌ Invalid format. Try again.", "31")
        today = datetime.date.today()
        start_date = (today - delta).isoformat()
        end_date = today.isoformat()
        while True:
            try:
                tas_threshold_input = input("\nEnter preliminary TAS threshold (0-10): ").strip()
                tas_threshold = float(tas_threshold_input)
                if 0 <= tas_threshold <= 10:
                    break
                else:
                    print_wrapped("❌ Enter a value between 0 and 10.", "31")
            except ValueError:
                print_wrapped("❌ Invalid input. Enter a numeric value.", "31")
        print_wrapped("⌛ Fetching CAD data...", "36")
        def fetch_and_store_data_concurrently(start_date: str, end_date: str, max_workers: int = 5):
            start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            current = start.replace(day=1)
            tasks = []
            while current <= end:
                year = current.year
                month = current.month
                month_start = current
                next_month = current + relativedelta(months=1)
                month_end = next_month - datetime.timedelta(days=1)
                if month_end > end:
                    month_end = end
                year_dir = os.path.join(CONFIG["DATA_DIR"], str(year))
                os.makedirs(year_dir, exist_ok=True)
                filename = f"{year}-{str(month).zfill(2)}.json"
                filepath = os.path.join(year_dir, filename)
                if os.path.exists(filepath):
                    current = next_month
                    continue
                tasks.append((year, month, month_start.isoformat(), month_end.isoformat()))
                current = next_month
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_info = {executor.submit(fetch_cad_data, ys, ye): (year, month) for (year, month, ys, ye) in tasks}
                with tqdm(total=len(future_to_info), desc="Fetching Data Sets", unit="set") as pbar:
                    for future in as_completed(future_to_info):
                        year, month = future_to_info[future]
                        try:
                            cad_data = future.result()
                            if cad_data and "data" in cad_data:
                                os.makedirs(os.path.join(CONFIG["DATA_DIR"], str(year)), exist_ok=True)
                                with open(os.path.join(CONFIG["DATA_DIR"], str(year), f"{year}-{str(month).zfill(2)}.json"), "w") as f:
                                    json.dump(cad_data, f, indent=4)
                            else:
                                logger.warning(f"No data for {year}-{str(month).zfill(2)}")
                        except Exception as e:
                            logger.error(f"Error for {year}-{str(month).zfill(2)}: {e}\n{traceback.format_exc()}")
                        pbar.update(1)
        fetch_and_store_data_concurrently(start_date, end_date)
        print_wrapped("✅ CAD data fetch complete.", "32")
        regions = define_geographic_regions()
        print_wrapped("⌛ Enriching and analyzing NEO data...", "36")
        data_usage = defaultdict(int)
        enriched_neos, raw_stats = analyze_data(regions, tas_threshold, start_date, end_date, data_usage, available_sources)
        print_wrapped(f"✅ Enriched {len(enriched_neos)} unique NEOs.", "32")
        with tqdm(total=1, desc="Performing Dynamic Analysis", position=3, unit="phase") as pbar_dyn:
            dynamic_category_counts, indicator_stats, dynamic_stats = dynamic_analysis(enriched_neos)
            pbar_dyn.update(1)
        print_wrapped("✅ Dynamic analysis complete.", "32")
        print("\nDynamic Category Counts:")
        for cat, cnt in dynamic_category_counts.items():
            print(f"  {cat}: {cnt}")
        save_results(enriched_neos, raw_stats, dynamic_category_counts, dynamic_stats, data_usage)
        print_beautified_console_summary(enriched_neos, raw_stats, dynamic_category_counts, dynamic_stats)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

