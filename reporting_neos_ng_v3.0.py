#!/usr/bin/env python3
"""
reporting_neos_ng_v3.0.py

NG Version of the aNEO Reporting System (v3.0) 🚀

This tool is designed for analyzing Near Earth Object (NEO) data. It loads raw and daily output data,
cleans and enriches it, applies AI‐driven anomaly detection, categorizes anomalies (marking them as Verified
or Unverified), ranks potential mission targets, and generates detailed reports and both 2D/3D visualizations.

Verified anomalies (marked as [Verified]) have an anomaly confidence > 10, indicating a significant deviation
from expected values and warranting further analysis. Unverified anomalies ([Unverified]) have lower confidence,
suggesting they may require additional review.
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, date
import traceback
import re
import time
import functools
import threading
import shutil
import textwrap

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Enable progress_apply for DataFrame.apply
tqdm.pandas()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "DATA_NEOS_DIR": "dataneos",
    "DAILY_OUTPUTS_DIR": os.path.join("dataneos", "daily_outputs"),
    "REPORTING_DIR": os.path.join("dataneos", "reporting"),
    "INPUT_FILE": "detailed_results.txt",  # fallback raw file if needed
    "PREPROCESSED_DATA_DIR": os.path.join("dataneos", "data"),
    "LOG_FILE": os.path.join("dataneos", f"reporting_{date.today().isoformat()}.log"),
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
        "velocity_shift": 5.0,
        "temporal_inertia": 100.0,
        "geo_eps": 5,
        "geo_min_samples": 2,
        "geo_min_clusters": 2,
        "diameter_min": 0.1,
        "diameter_max": 10.0,
        "albedo_min": 0.05,
        "albedo_max": 0.5,
        "min_subpoints": 2,
        "acceleration_threshold": 0.0005,
        "observation_gap_multiplier": 3,
        "albedo_artificial": 0.6
    },
    "REQUEST_TIMEOUT": 10,
    "MAX_RETRIES": 5,
    "INITIAL_RETRY_DELAY": 5
}

# ==============================================================================
# LOGGING SETUP WITH BEAUTIFIED SEPARATORS
# ==============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("reporting_neos")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=1_000_000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.info("\n" + "="*80 + "\n       START OF LOG - reporting_neos_ng_v3.0.py\n" + "="*80)
    return logger

def log_separator():
    logger.info("\n" + "-"*80)

# ==============================================================================
# INITIALIZE LOGGER
# ==============================================================================

logger = setup_logging()
log_separator()
logger.info("Initialized reporting_neos_ng_v3.0.py")

# ==============================================================================
# HELPER FUNCTIONS FOR TEXT WRAPPING
# ==============================================================================

def wrap_text(text: str) -> str:
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return textwrap.fill(text, width=width)

def print_wrapped(text: str, color_code: str = None):
    wrapped = wrap_text(text)
    if color_code:
        tqdm.write(colorize(wrapped, color_code))
    else:
        tqdm.write(wrapped)

def print_introduction():
    intro_text = (
        "Welcome to the aNEO Reporting System v3.0.\n\n"
        "This tool analyzes Near Earth Object (NEO) data by performing the following steps:\n"
        "  1. Loading raw and daily output data.\n"
        "  2. Enriching and deduplicating the data.\n"
        "  3. Handling incomplete data.\n"
        "  4. Applying legacy dynamic grouping and enhanced categorization.\n"
        "  5. Segmenting data into dynamic epochs.\n"
        "  6. Validating anomalies with an AI-driven model and filtering out slingshot effects.\n"
        "  7. Filtering to identify anomalous NEOs.\n"
        "  8. Ranking mission priority targets.\n"
        "  9. Generating detailed reports and visualizations (2D & 3D orbital maps).\n\n"
        "In the final summary, anomalies are marked as [Verified] if their anomaly confidence "
        "exceeds 10 (indicating a significant deviation from expected values), and as [Unverified] otherwise.\n"
        "Enjoy your analysis!"
    )
    print_wrapped(intro_text, color_code="36")
    tqdm.write(colorize("-" * 80, "35"))

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

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
    with tqdm(total=delay, desc=description, dynamic_ncols=True, leave=False,
              bar_format='{desc}: {n_fmt}/{total_fmt} seconds') as pbar:
        for _ in range(delay):
            time.sleep(1)
            pbar.update(1)

# ==============================================================================
# RAW DATA LOADING (Fallback)
# ==============================================================================

def load_raw_data() -> pd.DataFrame:
    df = pd.DataFrame()
    input_file_path = os.path.join(CONFIG["DATA_NEOS_DIR"], CONFIG["INPUT_FILE"])
    if os.path.exists(input_file_path):
        with open(input_file_path, "r") as f:
            content = f.read()
        parsed = parse_detailed_results(content)
        df = pd.DataFrame(parsed)
    return df

# ==============================================================================
# INCOMPLETE DATA HANDLING (Using .loc and explicit copying)
# ==============================================================================

def handle_incomplete_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expected_str_fields = ["Designation", "Observation Start", "Observation End", "Dynamic Category"]
    for field in expected_str_fields:
        if field not in df.columns:
            df.loc[:, field] = "unknown"
        else:
            df.loc[:, field] = df[field].fillna("unknown")
    numeric_fields = ["Raw TAS", "Dynamic TAS", "delta_v", "Close Approaches", "semi_major_axis", "eccentricity", "inclination"]
    for field in numeric_fields:
        if field not in df.columns:
            df.loc[:, field] = 0
        else:
            df.loc[:, field] = pd.to_numeric(df[field], errors="coerce")
            df.loc[:, field] = df[field].fillna(0)
    return df

def mark_incomplete_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "incomplete_data"] = False
    critical_fields = ["Designation", "Observation Start", "Observation End"]
    for field in critical_fields:
        df.loc[df[field] == "unknown", "incomplete_data"] = True
    return df

# ==============================================================================
# DATA PARSING FUNCTIONS (Using Strict Regex)
# ==============================================================================

@safe_execute
def parse_detailed_results(content: str) -> list:
    entries = []
    chunks = re.split(r"\n-{10,}\n", content)
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        record = {}
        for line in lines:
            line = line.strip()
            if not line or set(line) <= {"=", " "} or "DETAILED RESULTS" in line:
                continue
            m = re.match(r"^(.*?):\s*(.*)$", line)
            if m:
                key = m.group(1).strip()
                value = m.group(2).strip()
                if key in {"Raw TAS", "Dynamic TAS"}:
                    try:
                        record[key] = float(value)
                    except Exception:
                        record[key] = None
                elif key == "Close Approaches":
                    try:
                        record[key] = int(value)
                    except Exception:
                        record[key] = None
                elif key == "Observation Period":
                    parts = value.split(" to ")
                    if len(parts) == 2:
                        record["Observation Start"] = parts[0].strip()
                        record["Observation End"] = parts[1].strip()
                    else:
                        record[key] = value
                else:
                    record[key] = value
        if record.get("Designation"):
            entries.append(record)
    return entries

def load_detailed_results() -> pd.DataFrame:
    records = []
    if not os.path.isdir(CONFIG["DAILY_OUTPUTS_DIR"]):
        logger.warning(f"Daily outputs directory not found: {CONFIG['DAILY_OUTPUTS_DIR']}")
        return pd.DataFrame()
    file_paths = []
    for root, _, files in os.walk(CONFIG["DAILY_OUTPUTS_DIR"]):
        for file in files:
            if file == "detailed_results.txt":
                file_paths.append(os.path.join(root, file))
    if not file_paths:
        logger.warning("No detailed_results.txt files found in daily outputs.")
        return pd.DataFrame()
    for filepath in tqdm(file_paths, desc="Reading detailed_results.txt files", dynamic_ncols=True):
        try:
            with open(filepath, "r") as f:
                content = f.read()
            parsed = parse_detailed_results(content)
            if parsed:
                records.extend(parsed)
            logger.info(f"Parsed {len(parsed)} records from {filepath}.")
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    if records:
        return pd.DataFrame(records)
    else:
        logger.warning("No records parsed from daily outputs.")
        return pd.DataFrame()

# ==============================================================================
# DATA ENRICHMENT & DEDUPLICATION
# ==============================================================================

def enrich_and_deduplicate(df_daily: pd.DataFrame) -> pd.DataFrame:
    df_daily = df_daily.copy()
    df_raw = load_raw_data()
    if not df_raw.empty:
        df = pd.merge(df_daily, df_raw, on="Designation", how="outer", suffixes=("", "_raw"))
        required_fields = ["Observation Start", "Observation End", "Raw TAS", "Dynamic TAS", "delta_v", "Dynamic Category"]
        for field in required_fields:
            if field in df.columns and (field + "_raw") in df.columns:
                mask = (df[field].isnull()) | (df[field] == "unknown") | (((df[field] == 0) | (df[field] == 0.0)) & (df[field + "_raw"] != 0))
                df.loc[mask, field] = df.loc[mask, field + "_raw"]
        drop_cols = [col for col in df.columns if col.endswith("_raw")]
        df.drop(columns=drop_cols, inplace=True)
    else:
        df = df_daily.copy()
    df = df.drop_duplicates(subset=["Designation"]).copy()
    return df

# ==============================================================================
# DYNAMIC EPOCH SEGMENTATION
# ==============================================================================

def detect_epoch_shifts(df: pd.DataFrame) -> list:
    epoch_shifts = []
    if 'cluster_id' in df.columns and 'delta_v' in df.columns and 'observation_date' in df.columns:
        df = df.sort_values('observation_date').copy()
        cluster_changes = df['cluster_id'].diff().fillna(0).abs() > 0
        delta_v_change = df['delta_v'].diff().abs().fillna(0) > df['delta_v'].std()
        shift_points = df.loc[cluster_changes | delta_v_change, 'observation_date']
        epoch_shifts = sorted(shift_points.tolist())
    return epoch_shifts

def segment_by_dynamic_epochs(df: pd.DataFrame, epoch_shifts: list) -> dict:
    segments = {}
    if not epoch_shifts:
        segments['epoch_full'] = df.copy()
        return segments
    df = df.sort_values('observation_date').copy()
    start = df['observation_date'].min()
    for i, shift in enumerate(epoch_shifts):
        label = f"epoch_{i+1}"
        mask = (df['observation_date'] >= start) & (df['observation_date'] < shift)
        segments[label] = df.loc[mask].copy()
        start = shift
    segments[f"epoch_{len(epoch_shifts)+1}"] = df.loc[df['observation_date'] >= start].copy()
    return segments

# ==============================================================================
# LONG-TERM TRAJECTORY & PERSISTENT ANOMALY TRACKING
# ==============================================================================

def track_historical_trajectories(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'semi_major_axis' in df.columns and 'semi_major_axis' in history_df.columns:
        df = df.sort_values('observation_date').copy()
        df.loc[:, 'trajectory_deviation'] = df['semi_major_axis'].diff().abs().fillna(0)
    return df

def detect_orbital_deviations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'trajectory_deviation' in df.columns:
        threshold = df['trajectory_deviation'].mean() + df['trajectory_deviation'].std()
        df.loc[:, 'orbital_deviation_flag'] = df['trajectory_deviation'] > threshold
    return df

def track_persistent_anomalies(df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Designation' not in df.columns or 'Designation' not in previous_df.columns:
        return df
    df.loc[:, 'persistent_anomaly'] = df['Designation'].isin(previous_df['Designation'])
    if 'delta_v' in previous_df.columns:
        previous_delta = previous_df.set_index('Designation')['delta_v']
        df.loc[:, 'long_term_deviation'] = df.apply(
            lambda row: abs(row['delta_v'] - previous_delta.get(row['Designation'], row['delta_v'])),
            axis=1
        )
        df.loc[:, 'persistent_deviation'] = df['long_term_deviation'] > 0.1
    return df

# ==============================================================================
# AI-ASSISTED ORBITAL ANOMALY VALIDATION & SLINGSHOT DETECTION
# ==============================================================================

def run_with_spinner(func, *args, **kwargs):
    done = [False]
    def spin():
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while not done[0]:
            sys.stdout.write(f"\r{colorize('Training AI model... ' + spinner_chars[idx % len(spinner_chars)], '33')}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + colorize("Training AI model... done!          \n", "32"))
        sys.stdout.flush()
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()
    result = func(*args, **kwargs)
    done[0] = True
    spinner_thread.join()
    return result

def train_orbital_anomaly_model(df: pd.DataFrame) -> RandomForestRegressor:
    df = df.copy()
    features = ['semi_major_axis', 'eccentricity', 'inclination']
    for f in features:
        if f not in df.columns:
            logger.info(f"Feature '{f}' missing from data. Filling with default value 0.")
            df.loc[:, f] = 0.0
    X = df[features].fillna(0).values
    y = df['delta_v'].fillna(0).values
    model = run_with_spinner(RandomForestRegressor, n_estimators=100, random_state=42, verbose=0)
    model.fit(X, y)
    logger.info("Orbital anomaly model trained. ✅")
    return model

def validate_orbital_anomalies(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    df = df.copy()
    features = ['semi_major_axis', 'eccentricity', 'inclination']
    if not all(f in df.columns for f in features):
        return df
    if df[features].sum().sum() == 0:
         logger.warning("Orbital features are all zero; falling back to Dynamic TAS for anomaly detection.")
         if "Dynamic TAS" in df.columns and df["Dynamic TAS"].std() != 0:
             df.loc[:, 'anomaly_confidence'] = np.abs(df['Dynamic TAS'] - df['Dynamic TAS'].mean()) / (df['Dynamic TAS'].std() + 1e-6)
         else:
             df.loc[:, 'anomaly_confidence'] = 0
         df.loc[:, 'expected_delta_v'] = 0
    else:
         df.loc[:, 'expected_delta_v'] = model.predict(df[features].fillna(0).values)
         df.loc[:, 'anomaly_confidence'] = np.abs(df['delta_v'] - df['expected_delta_v']) / (df['expected_delta_v'] + 1e-6)
    dynamic_threshold = df['anomaly_confidence'].mean() + (df['anomaly_confidence'].std() * 1.5)
    df.loc[:, 'ai_validated_anomaly'] = df['anomaly_confidence'] >= dynamic_threshold
    df.loc[:, 'delta_v_anomaly_score'] = df['anomaly_confidence']
    return df

def detect_slingshot_effect(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta_v" in df.columns and df["delta_v"].std() != 0:
        df.loc[:, 'slingshot_flag'] = df['delta_v'].diff().abs() > (2 * df['delta_v'].std())
    else:
        df.loc[:, 'slingshot_flag'] = False
    return df

# ==============================================================================
# DISTANCE COMPUTATION FOR VISUALIZATION
# ==============================================================================

def compute_distance_from_earth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'semi_major_axis' in df.columns:
        df.loc[:, 'distance_from_earth'] = abs(df['semi_major_axis'] - 1.0)
    else:
        df.loc[:, 'distance_from_earth'] = 0
    return df

# ==============================================================================
# EARTH-CENTERED ORBITAL VISUALIZATION (At the End)
# ==============================================================================

def generate_orbital_map(df: pd.DataFrame, output_path: str):
    df = compute_distance_from_earth(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df['semi_major_axis'], df['inclination'], c=df['distance_from_earth'],
                    cmap='viridis', alpha=0.7, edgecolors='w', s=50)
    ax.set_xlabel("Semi-Major Axis (AU)")
    ax.set_ylabel("Inclination (Degrees)")
    avg_dist = df['distance_from_earth'].mean()
    ax.set_title(f"2D Orbital Map - Total aNEOs: {len(df)} | Avg Distance: {avg_dist:.2f} AU")
    cbar = fig.colorbar(sc)
    cbar.set_label("Distance from Earth (AU)")
    plt.savefig(output_path)
    plt.close()

def generate_3d_orbital_map(df: pd.DataFrame, output_path: str):
    df = compute_distance_from_earth(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['semi_major_axis'],
        y=df['eccentricity'],
        z=df['inclination'],
        mode='markers',
        marker=dict(size=3, color=df['distance_from_earth'], colorscale='Viridis', colorbar=dict(title="Dist (AU)")),
        name="aNEOs"
    ))
    fig.add_trace(go.Scatter3d(
        x=[1], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name="Earth"
    ))
    avg_dist = df['distance_from_earth'].mean()
    fig.update_layout(title=f"3D Orbital Map - Total aNEOs: {len(df)} | Avg Distance: {avg_dist:.2f} AU",
                      scene=dict(xaxis_title='Semi-Major Axis (AU)',
                                 yaxis_title='Eccentricity',
                                 zaxis_title='Inclination'))
    fig.write_html(output_path)

# ==============================================================================
# ADDITIONAL MECHANICS VERIFICATION FUNCTION
# ==============================================================================

def verify_mechanics(anomaly: dict) -> bool:
    # A verified anomaly is defined as one with anomaly confidence greater than 10.
    return anomaly.get("anomaly_confidence", 0) > 10

# ==============================================================================
# AUTOMATED CATEGORIZATION & REPORTING
# ==============================================================================

def escape_velocity(obj):
    return 11.2

def was_previously_neo(obj):
    return obj.get("previous_classification", "NEO") == "NEO"

def now_follows_standard_orbit(obj):
    return obj.get("eccentricity", 0) < CONFIG["THRESHOLDS"]["eccentricity"]

def categorize_object(obj, previous_classification):
    if obj.get("eccentricity", 0) > 1 and obj.get("v_inf", 0) > escape_velocity(obj):
        new_category = "ISO Candidate"
    elif obj.get("delta_v_anomaly_score", 0) > 2.0:
        new_category = "True Anomaly (ΔV)"
    elif was_previously_neo(obj) and now_follows_standard_orbit(obj):
        new_category = "NEO (Reclassified)"
    else:
        new_category = "NEO (Stable)"
    return new_category

def track_reclassification(obj, previous_classification):
    reclassification_reasons = []
    current_classification = obj.get("category", "N/A")
    if previous_classification == "NEO" and current_classification == "ISO Candidate":
        reclassification_reasons.append("Eccentricity exceeded 1 (Hyperbolic orbit).")
    if obj.get("previous_delta_v", 0) != obj.get("delta_v", 0):
        diff = abs(obj.get("previous_delta_v", 0) - obj.get("delta_v", 0))
        reclassification_reasons.append(f"ΔV changed by {diff:.2f} km/s.")
    if now_follows_standard_orbit(obj):
        reclassification_reasons.append("Now following a stable Keplerian orbit.")
    return reclassification_reasons

def categorize_aneos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, 'dynamic_category'] = 'Uncategorized'
    if 'cluster_id' in df.columns:
        df.loc[df['cluster_id'] != -1, 'dynamic_category'] = 'Permanent Orbital Cluster'
    if 'delta_v' in df.columns:
        threshold = df['delta_v'].std()
        df.loc[df['delta_v'] > threshold, 'dynamic_category'] = 'High ΔV/Maneuvering Object'
    if 'periodicity_score' in df.columns:
        median_val = df['periodicity_score'].median()
        df.loc[df['periodicity_score'] > median_val, 'dynamic_category'] = 'Synchronized Flyby'
    
    df.loc[:, "previous_classification"] = df["dynamic_category"]
    df.loc[:, "category"] = df.apply(lambda row: categorize_object(row, row["previous_classification"]), axis=1)
    df.loc[:, "reclassification_reasons"] = df.apply(lambda row: "; ".join(track_reclassification(row, row["previous_classification"])), axis=1)
    return df

def export_categorized_reports(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "categorized_aneos.csv"), index=False)

def generate_visual_reports(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "orbital_map.png")
    generate_orbital_map(df, map_path)

# ==============================================================================
# RANK MISSION PRIORITY TARGETS (UPDATED)
# ==============================================================================

def rank_mission_priority_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta_v_anomaly_score" not in df.columns and "anomaly_confidence" in df.columns:
        df.loc[:, "delta_v_anomaly_score"] = df["anomaly_confidence"]
    if "orbital_deviation_flag" in df.columns:
        df.loc[:, "orbital_perturbation_score"] = df["orbital_deviation_flag"].astype(int)
    else:
        df.loc[:, "orbital_perturbation_score"] = 0
    df.loc[:, "tisserand_classification_score"] = 0
    df.loc[:, "priority_score"] = (
        df["delta_v_anomaly_score"] * 2 +
        df["tisserand_classification_score"] * 1.5 +
        df["orbital_perturbation_score"]
    )
    return df.sort_values("priority_score", ascending=False)

# ==============================================================================
# REPORTING FUNCTIONS
# ==============================================================================

def save_results(anomalous_neos: list, raw_stats: dict, dynamic_category_counts: dict,
                 dynamic_stats: dict, data_usage: dict) -> None:
    base_output = os.path.join(CONFIG["REPORTING_DIR"],
                               f"aNEO_Report_{date.today().isoformat()}_{datetime.now().strftime('%H%M%S')}")
    os.makedirs(base_output, exist_ok=True)
    
    summary_filepath = os.path.join(base_output, "aNEO_Cluster_Report.txt")
    categorized_filepath = os.path.join(base_output, "aNEO_Categorized_Report.txt")
    priority_filepath = os.path.join(base_output, "aNEO_Mission_Priority.txt")
    anomaly_filepath = os.path.join(base_output, "aNEO_DeltaV_Anomaly_Report.txt")
    
    try:
        with open(summary_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("🚀 FINAL ANALYSIS REPORT 🚀\n")
            f.write("=" * 60 + "\n\n")
            f.write("Raw Analysis Statistics (Anomalous NEOs Only):\n")
            for key, value in raw_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nDynamic Category Counts:\n")
            for cat, cnt in dynamic_category_counts.items():
                f.write(f"  {cat}: {cnt}\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing cluster summary: {e}\n{traceback.format_exc()}")
    
    try:
        with open(categorized_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("📝 CATEGORIZED aNEO REPORT (Anomalous NEOs Only) 📝\n")
            f.write("=" * 60 + "\n\n")
            for neo in anomalous_neos:
                f.write(f"Designation: {neo.get('Designation', 'N/A')}\n")
                f.write(f"  Raw TAS: {neo.get('Raw TAS', 'N/A')}\n")
                f.write(f"  Dynamic TAS: {neo.get('Dynamic TAS', 'N/A')}\n")
                f.write(f"  Previous Classification: {neo.get('previous_classification', 'N/A')}\n")
                f.write(f"  New Category: {neo.get('category', 'N/A')}\n")
                if neo.get("reclassification_reasons"):
                    f.write(f"  Reclassification Reasons: {neo.get('reclassification_reasons')}\n")
                if neo.get("Observation Start") and neo.get("Observation End"):
                    f.write(f"  Observation Period: {neo['Observation Start']} to {neo['Observation End']}\n")
                f.write("  Full Raw Data:\n")
                for key, value in neo.items():
                    f.write(f"    {key}: {value}\n")
                f.write("-" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing categorized report: {e}\n{traceback.format_exc()}")
    
    try:
        with open(priority_filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("🚀 MISSION PRIORITY TARGETS (Anomalous NEOs Only) 🚀\n")
            f.write("=" * 60 + "\n\n")
            top_priority = sorted(anomalous_neos, key=lambda x: (x.get("priority_score") if x.get("priority_score") is not None else 0), reverse=True)
            for neo in top_priority:
                f.write(f"Designation: {neo.get('Designation', 'N/A')}\n")
                f.write(f"  Priority Score: {neo.get('priority_score', 'N/A')}\n")
                f.write(f"  Category: {neo.get('category', neo.get('dynamic_category', 'N/A'))}\n")
                f.write("-" * 60 + "\n")
    except Exception as e:
        logger.error(f"Error writing mission priority report: {e}\n{traceback.format_exc()}")
    
    try:
        with open(anomaly_filepath, "w") as f:
            header = "=" * 60 + "\n"
            f.write(header)
            f.write("🔥 ΔV ANOMALY REPORT (Anomalous NEOs Only) 🔥\n")
            f.write(header + "\n")
            f.write("DESIGNATION  |  ΔV (km/s)  |  Expected ΔV  |  Anomaly Confidence  |  Category\n")
            f.write("-" * 60 + "\n")
            for neo in anomalous_neos:
                if neo.get('ai_validated_anomaly'):
                    f.write(f"{neo.get('Designation', 'N/A'):12} | "
                            f"{neo.get('delta_v', 0):9.2f} | "
                            f"{neo.get('expected_delta_v', 0):11.2f} | "
                            f"{neo.get('anomaly_confidence', 0):18.2f} | "
                            f"{neo.get('category', 'N/A')}\n")
            f.write(header)
    except Exception as e:
        logger.error(f"Error writing anomaly report: {e}\n{traceback.format_exc()}")
    
    tqdm.write(colorize(f"\nReports have been generated in the directory: {base_output} 🎉", "32"))
    tqdm.write(colorize("Created reports:", "34"))
    tqdm.write("  - Cluster Report (aNEO_Cluster_Report.txt) 📊")
    tqdm.write("  - Categorized Report (aNEO_Categorized_Report.txt) 📝")
    tqdm.write("  - Mission Priority Report (aNEO_Mission_Priority.txt) 🚀")
    tqdm.write("  - ΔV Anomaly Report (aNEO_DeltaV_Anomaly_Report.txt) 🚨")

def print_beautified_console_summary(anomalous_neos: list, raw_stats: dict, dynamic_category_counts: dict, dynamic_stats: dict) -> None:
    border = "=" * 60
    tqdm.write(colorize(f"\n{border}", "35"))
    tqdm.write(colorize("🚀 FINAL ANALYSIS SUMMARY (Anomalous NEOs Only) 🚀".center(60), "33"))
    tqdm.write(colorize(f"{border}", "35"))
    tqdm.write(colorize(f"Total Anomalous NEOs: {raw_stats.get('Total Anomalous NEOs', 0)} 🚀", "32"))
    tqdm.write(colorize(f"Average Raw TAS: {raw_stats.get('Average Raw TAS', 0):.2f}", "32"))
    tqdm.write(colorize(f"Highest Raw TAS: {raw_stats.get('Highest Raw TAS', 0)}", "32"))
    tqdm.write(colorize(f"Lowest Raw TAS: {raw_stats.get('Lowest Raw TAS', 0)}", "32"))
    tqdm.write(colorize("\nDynamic Analysis:", "36"))
    tqdm.write(colorize(f"Highest Dynamic TAS: {raw_stats.get('Highest Dynamic TAS', 'N/A')}", "36"))
    tqdm.write(colorize(f"Lowest Dynamic TAS: {raw_stats.get('Lowest Dynamic TAS', 'N/A')}", "36"))
    tqdm.write(colorize("\nDynamic Category Counts:", "36"))
    for cat, cnt in dynamic_category_counts.items():
        tqdm.write(colorize(f"  {cat}: {cnt}", "36"))
    
    tqdm.write(colorize("\nTop 10 Mission Priority Anomalous NEOs:", "33"))
    top10 = sorted(anomalous_neos, key=lambda x: (x.get("priority_score") if x.get("priority_score") is not None else 0), reverse=True)[:10]
    for neo in top10:
        designation = neo.get("Designation", "N/A")
        priority = neo.get("priority_score", 0)
        verified_marker = "[Verified]" if verify_mechanics(neo) else "[Unverified]"
        tqdm.write(colorize(f"  {designation} - Priority Score: {priority} {verified_marker}", "32"))
    
    # Status Explanation (word-wrapped dynamically)
    status_text = (
        "Status Explanation:\n"
        "  [Verified]   : An anomaly is marked as verified if its anomaly confidence > 10, indicating a significant deviation "
        "from the expected ΔV. This suggests a genuine anomaly worthy of further analysis.\n"
        "  [Unverified] : An anomaly is marked as unverified if its anomaly confidence ≤ 10, suggesting a less pronounced deviation "
        "that may require further review."
    )
    print_wrapped(status_text, color_code="34")
    tqdm.write(colorize(f"{border}\n", "35"))

def print_anomaly_summary(anomalous_neos: list) -> None:
    top10 = sorted(anomalous_neos, key=lambda x: (x.get("anomaly_confidence") if x.get("anomaly_confidence") is not None else 0), reverse=True)[:10]
    header = colorize("=" * 60, "35")
    tqdm.write(header)
    tqdm.write(colorize("🔥 TOP 10 ΔV ANOMALY REPORT (Anomalous NEOs Only) 🔥".center(60), "33"))
    tqdm.write(header)
    tqdm.write(colorize("DESIGNATION  |  ΔV (km/s)  |  Expected ΔV  |  Anomaly Confidence  |  Category", "36"))
    tqdm.write("-" * 60)
    for neo in top10:
        if neo.get("ai_validated_anomaly"):
            verified_marker = "[Verified]" if verify_mechanics(neo) else "[Unverified]"
            line = (f"{neo.get('Designation', 'N/A'):12} | "
                    f"{neo.get('delta_v', 0):9.2f} | "
                    f"{neo.get('expected_delta_v', 0):11.2f} | "
                    f"{neo.get('anomaly_confidence', 0):18.2f} | "
                    f"{neo.get('category', 'N/A')} {verified_marker}")
            tqdm.write(colorize(line, "31"))
    tqdm.write(header)

# ==============================================================================
# AUTOMATED CATEGORIZATION & REPORTING - ADDITIONAL FUNCTIONS
# ==============================================================================

def escape_velocity(obj):
    return 11.2

def was_previously_neo(obj):
    return obj.get("previous_classification", "NEO") == "NEO"

def now_follows_standard_orbit(obj):
    return obj.get("eccentricity", 0) < CONFIG["THRESHOLDS"]["eccentricity"]

def categorize_object(obj, previous_classification):
    if obj.get("eccentricity", 0) > 1 and obj.get("v_inf", 0) > escape_velocity(obj):
        new_category = "ISO Candidate"
    elif obj.get("delta_v_anomaly_score", 0) > 2.0:
        new_category = "True Anomaly (ΔV)"
    elif was_previously_neo(obj) and now_follows_standard_orbit(obj):
        new_category = "NEO (Reclassified)"
    else:
        new_category = "NEO (Stable)"
    return new_category

def track_reclassification(obj, previous_classification):
    reclassification_reasons = []
    current_classification = obj.get("category", "N/A")
    if previous_classification == "NEO" and current_classification == "ISO Candidate":
        reclassification_reasons.append("Eccentricity exceeded 1 (Hyperbolic orbit).")
    if obj.get("previous_delta_v", 0) != obj.get("delta_v", 0):
        diff = abs(obj.get("previous_delta_v", 0) - obj.get("delta_v", 0))
        reclassification_reasons.append(f"ΔV changed by {diff:.2f} km/s.")
    if now_follows_standard_orbit(obj):
        reclassification_reasons.append("Now following a stable Keplerian orbit.")
    return reclassification_reasons

def categorize_aneos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, 'dynamic_category'] = 'Uncategorized'
    if 'cluster_id' in df.columns:
        df.loc[df['cluster_id'] != -1, 'dynamic_category'] = 'Permanent Orbital Cluster'
    if 'delta_v' in df.columns:
        threshold = df['delta_v'].std()
        df.loc[df['delta_v'] > threshold, 'dynamic_category'] = 'High ΔV/Maneuvering Object'
    if 'periodicity_score' in df.columns:
        median_val = df['periodicity_score'].median()
        df.loc[df['periodicity_score'] > median_val, 'dynamic_category'] = 'Synchronized Flyby'
    
    df.loc[:, "previous_classification"] = df["dynamic_category"]
    df.loc[:, "category"] = df.apply(lambda row: categorize_object(row, row["previous_classification"]), axis=1)
    df.loc[:, "reclassification_reasons"] = df.apply(lambda row: "; ".join(track_reclassification(row, row["previous_classification"])), axis=1)
    return df

def export_categorized_reports(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "categorized_aneos.csv"), index=False)

def generate_visual_reports(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "orbital_map.png")
    generate_orbital_map(df, map_path)

# ==============================================================================
# RANK MISSION PRIORITY TARGETS (UPDATED)
# ==============================================================================

def rank_mission_priority_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta_v_anomaly_score" not in df.columns and "anomaly_confidence" in df.columns:
        df.loc[:, "delta_v_anomaly_score"] = df["anomaly_confidence"]
    if "orbital_deviation_flag" in df.columns:
        df.loc[:, "orbital_perturbation_score"] = df["orbital_deviation_flag"].astype(int)
    else:
        df.loc[:, "orbital_perturbation_score"] = 0
    df.loc[:, "tisserand_classification_score"] = 0
    df.loc[:, "priority_score"] = (
        df["delta_v_anomaly_score"] * 2 +
        df["tisserand_classification_score"] * 1.5 +
        df["orbital_perturbation_score"]
    )
    return df.sort_values("priority_score", ascending=False)

# ==============================================================================
# MAIN EXECUTION FLOW (13 Steps)
# ==============================================================================

def main():
    # Print Introduction at the start (dynamically word-wrapped)
    print_introduction()
    
    meta_total_steps = 13
    meta_pbar = tqdm(total=meta_total_steps, desc="Overall Process", dynamic_ncols=True, position=0, leave=True)
    
    try:
        # ------- Step 1: Data Loading -------
        detailed_df = load_detailed_results()
        if detailed_df.empty:
            logger.warning("No detailed_results.txt files found; falling back to raw data.")
            detailed_df = load_raw_data()
        if detailed_df.empty:
            logger.error("No NEO data available after scanning. Exiting.")
            meta_pbar.close()
            return
        meta_pbar.update(1)
        
        # ------- Step 2: Data Enrichment & Deduplication -------
        df_merged = enrich_and_deduplicate(detailed_df)
        meta_pbar.update(1)
        
        # ------- Step 3: Incomplete Data Handling -------
        df_merged = handle_incomplete_data(df_merged)
        df_merged = mark_incomplete_data(df_merged)
        meta_pbar.update(1)
        
        # ------- Step 4: Legacy Dynamic Grouping -------
        enriched_neos = df_merged.to_dict(orient="records")
        for neo in enriched_neos:
            dyn_tas = neo.get("Dynamic TAS")
            dyn_cat = neo.get("Dynamic Category")
            if dyn_cat and dyn_cat.lower() != "unknown":
                neo["dynamic_category"] = dyn_cat
            elif dyn_tas is not None and dyn_tas != 0:
                if dyn_tas < 0.5:
                    neo["dynamic_category"] = "Within Normal Range"
                elif dyn_tas < 1.0:
                    neo["dynamic_category"] = "Slightly Anomalous"
                elif dyn_tas < 2.0:
                    neo["dynamic_category"] = "Moderately Anomalous"
                elif dyn_tas < 3.0:
                    neo["dynamic_category"] = "Highly Anomalous"
                else:
                    neo["dynamic_category"] = "Extremely Anomalous / Potentially Artificial"
            else:
                neo["dynamic_category"] = "Uncategorized"
        meta_pbar.update(1)
        
        # ------- Step 5: Enhanced Categorization -------
        for neo in enriched_neos:
            neo["previous_classification"] = neo.get("dynamic_category", "NEO")
            neo["category"] = categorize_object(neo, neo["previous_classification"])
            neo["reclassification_reasons"] = "; ".join(track_reclassification(neo, neo["previous_classification"]))
        meta_pbar.update(1)
        
        # ------- Step 6: Statistics & Dynamic Category Counts -------
        dynamic_category_counts = {}
        for neo in enriched_neos:
            cat = neo.get("dynamic_category", "Uncategorized")
            dynamic_category_counts[cat] = dynamic_category_counts.get(cat, 0) + 1
        raw_tas_values = [neo.get("Raw TAS") for neo in enriched_neos if neo.get("Raw TAS") not in [None, 0]]
        dynamic_tas_values = [neo.get("Dynamic TAS") for neo in enriched_neos if neo.get("Dynamic TAS") not in [None, 0]]
        raw_stats = {
            "Total NEOs Analyzed": len(enriched_neos),
            "Average Raw TAS": np.mean(raw_tas_values) if raw_tas_values else 0,
            "Highest Raw TAS": max(raw_tas_values) if raw_tas_values else 0,
            "Lowest Raw TAS": min(raw_tas_values) if raw_tas_values else 0,
            "Average Dynamic TAS": np.mean(dynamic_tas_values) if dynamic_tas_values else 0,
            "Highest Dynamic TAS": max(dynamic_tas_values) if dynamic_tas_values else 0,
            "Lowest Dynamic TAS": min(dynamic_tas_values) if dynamic_tas_values else 0,
        }
        top10 = sorted(enriched_neos, key=lambda x: x.get("Dynamic TAS", 0), reverse=True)[:10]
        dynamic_stats = {"Top 10 High Dynamic TAS NEOS": top10}
        meta_pbar.update(1)
        
        # ------- Step 7: Dynamic Epoch Segmentation -------
        if "Observation Start" in df_merged.columns:
            df_merged = df_merged.copy()
            df_merged.loc[:, "observation_date"] = pd.to_datetime(df_merged["Observation Start"], errors="coerce")
            epoch_shifts = detect_epoch_shifts(df_merged)
            epochs = segment_by_dynamic_epochs(df_merged, epoch_shifts)
            num_epochs = len(epochs)
            if num_epochs == 1:
                tqdm.write(colorize("Dynamic Epoch Segmentation: 1 epoch detected. ✅", "32"))
            else:
                tqdm.write(colorize(f"Dynamic Epoch Segmentation: {num_epochs} epochs detected. ✅", "32"))
        meta_pbar.update(1)
        
        # ------- Step 8: AI-Based Anomaly Validation & Slingshot Filtering -------
        anomaly_model = run_with_spinner(train_orbital_anomaly_model, df_merged)
        if anomaly_model is not None:
            df_merged = validate_orbital_anomalies(df_merged, anomaly_model)
            df_merged = detect_slingshot_effect(df_merged)
            df_merged.loc[:, 'ai_validated_anomaly'] = df_merged['ai_validated_anomaly'] & (~df_merged['slingshot_flag'])
            enriched_neos = df_merged.to_dict(orient="records")
            tqdm.write(colorize("AI-Based ΔV anomaly validation and slingshot filtering complete! 🤖", "32"))
        meta_pbar.update(1)
        
        # ------- Step 9: Filtering Anomalous NEOs -------
        anomalous_neos = [neo for neo in enriched_neos if 
                          neo.get("ai_validated_anomaly", False) or 
                          (neo.get("category") == "ISO Candidate") or 
                          (neo.get("delta_v_anomaly_score", 0) > 1.5)]
        if anomalous_neos:
            raw_tas_values = [neo.get("Raw TAS") for neo in anomalous_neos if neo.get("Raw TAS") not in [None, 0]]
            dynamic_tas_values = [neo.get("Dynamic TAS") for neo in anomalous_neos if neo.get("Dynamic TAS") not in [None, 0]]
            raw_stats = {
                "Total Anomalous NEOs": len(anomalous_neos),
                "Average Raw TAS": np.mean(raw_tas_values) if raw_tas_values else 0,
                "Highest Raw TAS": max(raw_tas_values) if raw_tas_values else 0,
                "Lowest Raw TAS": min(raw_tas_values) if raw_tas_values else 0,
                "Average Dynamic TAS": np.mean(dynamic_tas_values) if dynamic_tas_values else 0,
                "Highest Dynamic TAS": max(dynamic_tas_values) if dynamic_tas_values else 0,
                "Lowest Dynamic TAS": min(dynamic_tas_values) if dynamic_tas_values else 0,
            }
            dynamic_category_counts = {}
            for neo in anomalous_neos:
                cat = neo.get("dynamic_category", "Uncategorized")
                dynamic_category_counts[cat] = dynamic_category_counts.get(cat, 0) + 1
        else:
            tqdm.write(colorize("No anomalous NEOs detected.", "31"))
            raw_stats = {}
            dynamic_category_counts = {}
        meta_pbar.update(1)
        
        # ------- Step 10: Mission Priority Ranking -------
        ranked_df = rank_mission_priority_targets(df_merged)
        ranked_neos = ranked_df.to_dict(orient="records")
        for neo in enriched_neos:
            for ranked in ranked_neos:
                if ranked.get("Designation") == neo.get("Designation"):
                    neo["priority_score"] = ranked.get("priority_score")
                    break
        meta_pbar.update(1)
        
        # ------- Step 11: Saving Reports -------
        save_results(anomalous_neos, raw_stats, dynamic_category_counts, dynamic_stats, data_usage={})
        meta_pbar.update(1)
        
        # ------- Step 12: Generating Visualizations -------
        df_merged = compute_distance_from_earth(df_merged)
        orb2d_path = os.path.join(CONFIG["REPORTING_DIR"], f"orbital_map_{datetime.now().strftime('%H%M%S')}.png")
        orb3d_path = os.path.join(CONFIG["REPORTING_DIR"], f"orbital_map_3d_{datetime.now().strftime('%H%M%S')}.html")
        generate_orbital_map(df_merged, orb2d_path)
        generate_3d_orbital_map(df_merged, orb3d_path)
        tqdm.write(colorize(f"2D orbital map saved to {orb2d_path}. 🖼️", "32"))
        tqdm.write(colorize(f"3D orbital map saved to {orb3d_path}. 🖥️", "32"))
        meta_pbar.update(1)
        
        # ------- Step 13: Console Summary -------
        print_beautified_console_summary(anomalous_neos, raw_stats, dynamic_category_counts, dynamic_stats)
        print_anomaly_summary(anomalous_neos)
        top10_distance = sorted(anomalous_neos, key=lambda x: x.get("distance_from_earth", 0) if x.get("distance_from_earth") is not None else 0, reverse=True)[:10]
        tqdm.write(colorize("\nTop 10 aNEOs by Distance from Earth:", "33"))
        for neo in top10_distance:
            dist = neo.get("distance_from_earth", 0)
            designation = neo.get("Designation", "N/A")
            tqdm.write(colorize(f"  {designation} - Distance: {dist:.2f} AU", "32"))
        meta_pbar.update(1)
        
        meta_pbar.close()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        meta_pbar.close()
        sys.exit(1)
 
if __name__ == "__main__":
    main()

