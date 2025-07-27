# src/data_processing/cleaner.py

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on the earth (in km)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c # Radius of earth in kilometers

def _extract_coords(point_str):
    """Extracts longitude and latitude from a POINT string like 'POINT(lon lat)'."""
    if not isinstance(point_str, str):
        return (None, None)
    # Updated regex to be more robust to whitespace variations
    match = re.search(r'POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)', point_str)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except (ValueError, AttributeError):
            return (None, None)
    return (None, None)

def process_and_clean_orders(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive cleaning pipeline for the order data.
    Returns the cleaned dataframe and a report of the cleaning process.
    """
    report = {}
    initial_rows = len(df)
    report['initial_rows'] = initial_rows
    
    # Handle Critical Missing Values first
    df.dropna(subset=['trip_start', 'trip_end', 'start_location', 'end_location', 'license_plate', 'fare'], inplace=True)
    rows_after_dropna = len(df)
    report['critical_na_removed'] = initial_rows - rows_after_dropna
    
    # --- Coordinate Parsing ---
    logger.info("Parsing coordinates from location strings...")
    for stage in ['start', 'end']:
        loc_col = f'{stage}_location'
        lon_col = f'{stage}_lon'
        lat_col = f'{stage}_lat'
        if loc_col in df.columns:
            coords = df[loc_col].apply(_extract_coords)
            df[lon_col] = coords.apply(lambda x: x[0])
            df[lat_col] = coords.apply(lambda x: x[1])
    
    df.dropna(subset=['start_lon', 'start_lat', 'end_lon', 'end_lat'], inplace=True)
    
    # --- Timestamp Parsing and Timezone Conversion ---
    logger.info("Parsing timestamps to UTC, then converting to Beijing Time (Asia/Shanghai)...")
    df['trip_start'] = pd.to_datetime(df['trip_start'], utc=True, errors='coerce')
    df['trip_end'] = pd.to_datetime(df['trip_end'], utc=True, errors='coerce')
    df.dropna(subset=['trip_start', 'trip_end'], inplace=True)

    beijing_tz = 'Asia/Shanghai'
    df['trip_start'] = df['trip_start'].dt.tz_convert(beijing_tz)
    df['trip_end'] = df['trip_end'].dt.tz_convert(beijing_tz)

    rows_after_parse = len(df)
    report['parse_failures_removed'] = rows_after_dropna - rows_after_parse
    
    # --- Filter by Bounding Box ---
    if 'spatio_temporal' in config and 'city_bounding_box' in config['spatio_temporal']:
        bbox = config['spatio_temporal']['city_bounding_box']
        df = df[(df['start_lon'] >= bbox[0]) & (df['start_lon'] <= bbox[2]) &
                (df['start_lat'] >= bbox[1]) & (df['start_lat'] <= bbox[3])]
        df = df[(df['end_lon'] >= bbox[0]) & (df['end_lon'] <= bbox[2]) &
                (df['end_lat'] >= bbox[1]) & (df['end_lat'] <= bbox[3])]
    rows_after_bbox = len(df)
    report['geo_outliers_removed'] = rows_after_parse - rows_after_bbox
    
    # --- Filter by Trip Attributes ---
    df['duration_hours'] = (df['trip_end'] - df['trip_start']).dt.total_seconds() / 3600.0
    df['implied_speed_kmh'] = (df['distance'] / 1000.0) / (df['duration_hours'] + 1e-6)

    df = df[df['duration_hours'].between(1/60, 5)]
    df = df[df['distance'].between(50, 150000)]
    df = df[df['implied_speed_kmh'] <= 150]
    df = df[df['fare'] <= 2000]
    rows_after_logical_filter = len(df)
    report['logical_outliers_removed'] = rows_after_bbox - rows_after_logical_filter
    
    # --- Final Selection ---
    final_cols = ['id', 'license_plate', 'trip_start', 'trip_end', 'start_lon', 'start_lat', 'end_lon', 'end_lat', 'distance', 'fare', 'duration_hours', 'implied_speed_kmh']
    df_clean = df[[col for col in final_cols if col in df.columns]].copy()
    
    report['final_rows'] = len(df_clean)
    report['total_removed'] = initial_rows - len(df_clean)
    report['removed_percentage'] = f"{(report['total_removed'] / initial_rows * 100):.2f}%"
    
    logger.info(f"Order data cleaning report: {report}")
    return df_clean, report

def process_and_clean_gps(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """Processes and cleans the raw GPS data."""
    report = {}
    initial_rows = len(df)
    report['initial_rows'] = initial_rows

    # Rename columns for consistency
    df.rename(columns={'createdAt': 'device_time', 'f0_': 'vehicle_id', 'location': 'point'}, inplace=True)

    # Parse coordinates
    coords = df['point'].apply(_extract_coords)
    df['lon'] = coords.apply(lambda x: x[0])
    df['lat'] = coords.apply(lambda x: x[1])
    df.dropna(subset=['lon', 'lat'], inplace=True)

    # Parse timestamps
    df['device_time'] = pd.to_datetime(df['device_time'], utc=True, errors='coerce')
    df.dropna(subset=['device_time'], inplace=True)
    df['device_time'] = df['device_time'].dt.tz_convert('Asia/Shanghai')

    # Filter by bounding box
    if 'spatio_temporal' in config and 'city_bounding_box' in config['spatio_temporal']:
        bbox = config['spatio_temporal']['city_bounding_box']
        df = df[(df['lon'] >= bbox[0]) & (df['lon'] <= bbox[2]) &
                (df['lat'] >= bbox[1]) & (df['lat'] <= bbox[3])]

    final_cols = ['vehicle_id', 'device_time', 'lon', 'lat']
    df_clean = df[[col for col in final_cols if col in df.columns]].copy()

    report['final_rows'] = len(df_clean)
    report['total_removed'] = initial_rows - len(df_clean)
    report['removed_percentage'] = f"{(report['total_removed'] / initial_rows * 100):.2f}%"

    logger.info(f"GPS data cleaning report: {report}")
    return df_clean, report