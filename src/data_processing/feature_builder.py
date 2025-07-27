# src/data_processing/feature_builder.py

import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path # Import Path
from .gridding import BaseMapper
from ..utils.common import save_dataframe # Import save_dataframe
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

def infer_vehicle_status(df_gps: pd.DataFrame, df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Infers the status (0 for free, 1 for occupied) for each GPS point.
    (Logic remains the same as before)
    """
    # --- Guard Clause for Empty GPS Data ---
    if df_gps.empty:
        logging.info("GPS data is empty, skipping vehicle status inference.")
        return pd.DataFrame(columns=['lon', 'lat', 'device_time', 'status'])

    logging.info("Inferring vehicle status for each GPS point... (This may take a while)")
    
    df_gps_sorted = df_gps.sort_values(by=['vehicle_id', 'device_time'])
    df_orders_sorted = df_orders.sort_values(by=['license_plate', 'trip_start'])
    
    occupied_windows = {}
    for vehicle_id, trips in df_orders_sorted.groupby('license_plate'):
        occupied_windows[vehicle_id] = list(zip(trips['trip_start'], trips['trip_end']))

    def get_status(row: pd.Series) -> int:
        vehicle_id = row['vehicle_id']
        timestamp = row['device_time']
        
        if vehicle_id not in occupied_windows:
            return 0 
        
        for start, end in occupied_windows[vehicle_id]:
            if start <= timestamp <= end:
                return 1 
        return 0 
    
    df_gps_with_status = df_gps_sorted.copy() # Work on a copy
    df_gps_with_status['status'] = df_gps_with_status.progress_apply(get_status, axis=1)
    logging.info("Finished inferring vehicle status.")
    logging.info(f"Status distribution:\n{df_gps_with_status['status'].value_counts(normalize=True)}")
    
    return df_gps_with_status

def add_advanced_interaction_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    根据业务理解创建丰富多样的交叉特征，以帮助模型捕捉复杂的非线性关系。
    
    Args:
        df (pd.DataFrame): 已经添加了基础特征（时间、天气、静态等）的数据框。
        config (dict): 配置文件。

    Returns:
        pd.DataFrame: 添加了交叉特征后的数据框。
    """
    logging.info("正在创建高级交叉特征...")
    
    # ==========================================================================
    # 类别 1: 时间周期交互 (Time x Time Period Interactions)
    # 目标：让模型理解不同小时在一天中不同时段、工作日/周末的影响差异。
    # ==========================================================================
    if 'hour' in df.columns and 'is_weekend' in df.columns:
        # 创建一个“周末激活”的小时特征
        df['hour_x_is_weekend'] = df['hour'] * df['is_weekend']
        logging.info("  [Time x Period] 已创建特征: hour_x_is_weekend")
        
        # 定义一天中的不同时段 
        bins = [-1, 5, 10, 16, 20, 23] # [深夜, 早高峰, 日间, 晚高峰, 夜间]
        labels = ['late_night', 'morning_peak', 'daytime', 'evening_peak', 'night']
        df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels)
        
        # 将时段特征进行独热编码 (One-Hot Encoding)
        time_period_dummies = pd.get_dummies(df['time_period'], prefix='period')
        df = pd.concat([df, time_period_dummies], axis=1)
        logging.info(f"  [Time x Period] 已创建独热编码时段特征: {list(time_period_dummies.columns)}")
        
        # 小时与具体时段的交互
        for period in labels:
            col_name = f'period_{period}'
            if col_name in df.columns:
                df[f'hour_x_{col_name}'] = df['hour'] * df[col_name]
        logging.info("  [Time x Period] 已创建小时与具体时段的交叉特征。")


    # ==========================================================================
    # 类别 2: 天气与时间交互 (Weather x Time Interactions)
    # 目标：捕捉天气在不同时间点对需求的影响。
    # ==========================================================================
    if 'precipitation' in df.columns and 'hour' in df.columns:
        # 定义一个“高峰时段”的标志
        peak_hours = config.get('features', {}).get('peak_hours', [7, 8, 9, 17, 18, 19])
        df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)
        
        # 降雨在高峰时段的影响可能更大
        df['peak_hour_x_precipitation'] = df['is_peak_hour'] * df['precipitation']
        logging.info("  [Weather x Time] 已创建特征: peak_hour_x_precipitation")
        
    if 'temperature_2m' in df.columns and 'hour_sin' in df.columns:
        # 温度与周期性时间特征的交互
        df['temp_x_hour_sin'] = df['temperature_2m'] * df['hour_sin']
        df['temp_x_hour_cos'] = df['temperature_2m'] * df['hour_cos']
        logging.info("  [Weather x Time] 已创建特征: temp_x_hour_sin/cos")

    # ==========================================================================
    # 类别 3: 空间与时间交互 (Spatial x Time Interactions)
    # 目标：让模型理解一个区域的特性在不同时间如何影响需求。
    # ==========================================================================
    # 假设 't_pop' (总人口) 和 't_wp' (工作人口) 是静态特征
    if 't_pop' in df.columns and 't_wp' in df.columns and 'is_weekend' in df.columns:
        # 创建一个“工作日通勤时段”的标志
        df['is_weekday_commute'] = ((df['is_weekend'] == 0) & df['hour'].isin([7,8,9,17,18,19])).astype(int)
        
        # “活跃通勤人口”：一个动态特征，只在工作日通勤时段反映该区域的工作人口数
        df['active_commute_population'] = df['t_wp'] * df['is_weekday_commute']
        logging.info("  [Spatial x Time] 已创建特征: active_commute_population")

    # 假设 'ma_hh' (家庭收入中位数) 是静态特征
    if 'ma_hh' in df.columns and 'is_weekend' in df.columns:
        # 收入水平与周末的交互，可能高收入区在周末有更多休闲出行需求
        df['income_x_is_weekend'] = df['ma_hh'] * df['is_weekend']
        logging.info("  [Spatial x Time] 已创建特征: income_x_is_weekend")
        
    # ==========================================================================
    # 类别 4: 历史趋势与当前时间交互 (History x Time Interactions)
    # 目标：让模型理解历史趋势在不同时间点的重要性。
    # ==========================================================================
    # 假设 'demand_roll_1h_mean' 是创建的1小时滑动平均需求
    if 'demand_roll_1h_mean' in df.columns and 'hour' in df.columns:
        # 近期需求趋势与小时的交互
        df['recent_demand_x_hour'] = df['demand_roll_1h_mean'] * df['hour']
        logging.info("  [History x Time] 已创建特征: recent_demand_x_hour")
    
    # 滞后特征与当前供给的交互
    # 假设 'demand_lag_daily' 是日滞后需求
    if 'demand_lag_daily' in df.columns and 'supply' in df.columns:
        # 如果昨天这个时候需求很高，但现在车很多，可能意味着机会减少
        df['daily_lag_demand_div_supply'] = df['demand_lag_daily'] / (df['supply'] + 1e-6)
        logging.info("  [History x Current] 已创建特征: daily_lag_demand_div_supply")

    # 清理掉中间创建的辅助列
    df.drop(columns=['time_period', 'is_peak_hour', 'is_weekday_commute'], inplace=True, errors='ignore')

    logging.info("已完成高级交叉特征的创建。")
    return df

def add_time_features(df: pd.DataFrame, time_col: str = 'time_slot', config: dict = None) -> pd.DataFrame:
    """Adds time-based features to the DataFrame."""
    if not config or not config['features']['enable_time_features']:
        logging.info("Time features are disabled in config.")
        return df

    logging.info(f"Adding time features based on column: {time_col}")
    dt_series = pd.to_datetime(df[time_col])
    
    df['hour'] = dt_series.dt.hour
    df['dayofweek'] = dt_series.dt.dayofweek # Monday=0, Sunday=6
    df['dayofyear'] = dt_series.dt.dayofyear
    df['month'] = dt_series.dt.month
    df['weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Sine/Cosine encoding for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    if config['features'].get('holidays_list'):
        holidays = pd.to_datetime(config['features']['holidays_list']).date
        df['is_holiday'] = dt_series.dt.date.isin(holidays).astype(int)
    else:
        df['is_holiday'] = 0
        
    logging.info("Finished adding time features.")
    return df

def add_lagged_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adds lagged features (e.g., demand/supply from previous day/week)."""
    if not config.get('features', {}).get('enable_lagged_features', False):
        logging.info("Lagged features are disabled in config.")
        return df

    lag_config = config.get('features', {}).get('lag_features_to_create')
    if not lag_config:
        logging.info("'lag_features_to_create' not found in config. Skipping lagged features.")
        return df

    logging.info("Adding lagged features...")
    # Ensure DataFrame is sorted by zone_id and time_slot for correct lagging
    df = df.sort_values(by=['zone_id', 'time_slot'])
    
    time_interval_min = config['spatio_temporal']['time_interval_minutes']
    periods_per_day = (24 * 60) // time_interval_min
    periods_per_week = (7 * 24 * 60) // time_interval_min
    
    lag_config = config['features']['lag_features_to_create']
    
    for target_col, lags in lag_config.items():
        if target_col not in df.columns:
            logging.warning(f"Target column '{target_col}' for lagging not found in DataFrame. Skipping.")
            continue
        if lags.get('daily'):
            lag_col_name = f'{target_col}_lag_daily'
            df[lag_col_name] = df.groupby('zone_id')[target_col].shift(periods_per_day)
            logging.info(f"Added daily lag for {target_col} as {lag_col_name}")
        if lags.get('weekly'):
            lag_col_name = f'{target_col}_lag_weekly'
            df[lag_col_name] = df.groupby('zone_id')[target_col].shift(periods_per_week)
            logging.info(f"Added weekly lag for {target_col} as {lag_col_name}")
            
    logging.info("Finished adding lagged features.")
    return df

def add_rolling_window_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adds rolling window statistical features."""
    if not config.get('features', {}).get('enable_rolling_window_features', False):
        logging.info("Rolling window features are disabled in config.")
        return df

    window_hours = config.get('features', {}).get('rolling_window_hours')
    if not window_hours:
        logging.info("'rolling_window_hours' not found in config. Skipping rolling window features.")
        return df

    logging.info("Adding rolling window features...")
    df = df.sort_values(by=['zone_id', 'time_slot'])
    
    time_interval_min = config['spatio_temporal']['time_interval_minutes']
    window_hours = config['features']['rolling_window_hours']
    target_cols = config['features']['rolling_window_target_cols']
    stats_to_calc = config['features']['rolling_window_stats']
    
    for col in target_cols:
        if col not in df.columns:
            logging.warning(f"Target column '{col}' for rolling window not found. Skipping.")
            continue
        for wh in window_hours:
            window_size = (wh * 60) // time_interval_min
            if window_size < 1:
                logging.warning(f"Rolling window size for {wh}h is less than 1 period. Skipping.")
                continue
            
            # min_periods=1 ensures that we get a value even if the window is not full (e.g. at the beginning of series)
            # Group by zone_id, then apply rolling
            grouped = df.groupby('zone_id')[col]
            
            for stat in stats_to_calc:
                roll_col_name = f'{col}_roll_{wh}h_{stat}'
                if stat == 'mean':
                    df[roll_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
                elif stat == 'median':
                    df[roll_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).median())
                elif stat == 'std':
                    df[roll_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
                elif stat == 'min':
                    df[roll_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).min())
                elif stat == 'max':
                    df[roll_col_name] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).max())
                logging.info(f"Added rolling {stat} for {col} over {wh}h as {roll_col_name}")
                
    logging.info("Finished adding rolling window features.")
    return df

def add_weather_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Loads weather data and merges it with the main feature DataFrame."""
    if not config['features']['enable_weather_features']:
        logging.info("Weather features are disabled in config.")
        return df

    logging.info("Adding weather features...")
    weather_data_path = config['data'].get('weather_data_path')
    if not weather_data_path or not Path(weather_data_path).exists():
        logging.warning(f"Weather data file not found at {weather_data_path} or path not specified. Skipping weather features.")
        return df

    try:
        weather_df = pd.read_csv(weather_data_path)
        logging.info(f"Loaded weather data from {weather_data_path} with {len(weather_df)} rows.")
    except Exception as e:
        logging.error(f"Error loading weather data: {e}. Skipping weather features.")
        return df

    # Preprocess weather data
    weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True) # Assuming 'time' column exists
    
    # Align weather time to our feature DataFrame's time_slot (which is e.g., 15-min interval)
    # We can round weather time to the nearest hour, then merge.
    # Or, more robustly, merge based on 'time_slot' hour and 'zone_id'.
    
    # Create 'hour_start_time' in our main df for merging
    df['hour_start_time'] = pd.to_datetime(df['time_slot']).dt.floor('h')
    
    # Prepare weather_df for merge:
    # Ensure 'location' in weather_df matches our 'zone_id' concept
    weather_location_col = 'location' # As per user's weather data header
    if config['features']['weather_location_col_is_zone_id']:
        weather_df.rename(columns={weather_location_col: 'zone_id'}, inplace=True)
    else:
        # If weather location is coordinates, map it to zone_id
        # This part requires weather_df to have 'lon', 'lat' columns or similar
        logging.warning("Weather location is not zone_id. Geo-mapping needed but not implemented yet. Skipping weather.")
        return df # Or implement geo-mapping if weather_df has coords

    weather_df.rename(columns={'time': 'hour_start_time'}, inplace=True)
    
    weather_feature_cols = config['features']['weather_feature_cols']
    cols_to_merge = ['hour_start_time', 'zone_id'] + weather_feature_cols
    
    if not all(col in weather_df.columns for col in cols_to_merge if col not in ['hour_start_time', 'zone_id']):
        logging.error(f"Missing one or more weather_feature_cols in weather_df. Available: {weather_df.columns}. Needed: {weather_feature_cols}")
        return df

    # Merge weather data
    df = pd.merge(df, weather_df[cols_to_merge], on=['hour_start_time', 'zone_id'], how='left')
    
    # Handle missing weather data after merge (e.g., forward fill per zone)
    for col in weather_feature_cols:
        if col in df.columns:
            # Forward fill then backward fill within each zone
            df[col] = df.groupby('zone_id')[col].ffill().bfill()
            # Fill any remaining NaNs (e.g., a zone with no weather data at all) with mean or 0
            df[col] = df[col].fillna(df[col].mean()) # Or fill with 0, or a specific value
            logging.info(f"Merged and imputed weather feature: {col}")
        else:
            logging.warning(f"Weather feature {col} not found after merge.")

    df.drop(columns=['hour_start_time'], inplace=True, errors='ignore')
    logging.info("Finished adding weather features.")
    return df


# --- NEW FUNCTION for Static Zone Features ---

def add_static_zone_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Loads static zone-based features from an Excel file, cleans them, and merges them.
    """
    if not config.get('features', {}).get('enable_static_zone_features', False):
        logging.info("Static zone features are disabled in config.")
        return df

    logging.info("Adding static zone features...")
    static_features_path = config['data'].get('zone_static_features_path')
    if not static_features_path or not Path(static_features_path).exists():
        logging.warning(f"Static zone features file not found at {static_features_path}. Skipping.")
        return df

    try:
        # Use read_excel for .xlsx files
        static_df = pd.read_excel(static_features_path)
        logging.info(f"Loaded static features from {static_features_path}")
    except Exception as e:
        logging.error(f"Error loading static features Excel file: {e}. Skipping.")
        return df

    # --- RENAME COLUMNS TO MATCH CONFIG ABBREVIATIONS ---
    rename_map = config.get('features', {}).get('static_zone_feature_cols_rename_map', {})
    if rename_map:
        static_df.rename(columns=rename_map, inplace=True)
        logging.info("Renamed static feature columns based on config map.")

    zone_id_col = config['features']['static_zone_id_col']
    feature_cols = config['features']['static_zone_feature_cols']

    if zone_id_col not in static_df.columns:
        logging.error(f"Zone ID column ''{zone_id_col}'' not found in static features file. Skipping.")
        return df

    # Filter for only the required columns
    valid_cols = [col for col in feature_cols if col in static_df.columns]
    missing_cols = set(feature_cols) - set(valid_cols)
    if missing_cols:
        logging.warning(f"Static features specified but not found in file: {missing_cols}")
    
    if not valid_cols:
        logging.warning("No valid static features found to load. Skipping.")
        return df

    static_df = static_df[[zone_id_col] + valid_cols]

    # Robust data cleaning for selected columns
    for col in valid_cols:
        # If column is object type, it might contain commas for thousands
        if static_df[col].dtype == 'object':
            static_df[col] = static_df[col].astype(str).str.replace(',', '', regex=False)
        
        # Convert to numeric, coercing errors to NaN
        static_df[col] = pd.to_numeric(static_df[col], errors='coerce')

        # Impute any NaNs with the column median for robustness
        if static_df[col].isnull().any():
            median_val = static_df[col].median()
            static_df[col].fillna(median_val, inplace=True)
            logging.info(f"Cleaned and imputed NaNs in static feature ''{col}'' with median value {median_val}.")

    # Rename the ID column for merging
    static_df.rename(columns={zone_id_col: 'zone_id'}, inplace=True)

    # ==> FIX: Enforce consistent string data type on the join key
    df['zone_id'] = df['zone_id'].astype(str)
    static_df['zone_id'] = static_df['zone_id'].astype(str)

    # Merge into the main dataframe
    df = pd.merge(df, static_df, on='zone_id', how='left')
    logging.info(f"Successfully merged {len(valid_cols)} static features.")

    return df


# --- Main Feature Building Function ---

def build_spatio_temporal_features(
    df_orders_clean: pd.DataFrame, 
    df_gps: pd.DataFrame, 
    mapper: BaseMapper, 
    time_interval_min: int,
    config: dict
) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline for the three-target decoupled model.
    """
    # --- Part 1: Data Preparation ---
    logging.info("Preparing data for the three-target model...")
    
    df_orders_clean.rename(columns={'id': 'order_id'}, inplace=True)
    origin_zones = mapper.assign_zone_id(df_orders_clean.copy(), lon_col='start_lon', lat_col='start_lat')
    df_orders_final = pd.merge(df_orders_clean, origin_zones[['order_id', 'zone_id']], on='order_id', how='inner')
    df_orders_final.rename(columns={'zone_id': 'origin_zone_id'}, inplace=True)
    logging.info(f"Orders after origin zone assignment: {df_orders_final['origin_zone_id'].nunique()} unique zones.")
    
    time_res = f"{time_interval_min}min"
    df_orders_final['time_slot'] = df_orders_final['trip_start'].dt.floor(time_res)
    
    # --- Part 2: Aggregate base features for the three-target model ---
    logging.info("Aggregating base features: trip_count and average_income...")
    
    demand_income_agg = df_orders_final.groupby(['time_slot', 'origin_zone_id']).agg(
        trip_count=('order_id', 'size'),
        average_income=('fare', 'mean'),
        total_income=('fare', 'sum')  # Add the true total income
    ).reset_index().rename(columns={'origin_zone_id': 'zone_id'})
    logging.info(f"Demand/Income aggregation unique zones: {demand_income_agg['zone_id'].nunique()}")

    # Aggregate supply data
    intermediate_dir = Path(config['data']['intermediate_dir'])
    path_gps_with_status = intermediate_dir / config['data']['gps_with_status_filename']
    if config['data']['use_intermediate_files'] and path_gps_with_status.exists():
        df_gps_for_zoning = pd.read_pickle(path_gps_with_status)
    else:
        df_gps_for_zoning = infer_vehicle_status(df_gps.copy(), df_orders_clean.copy())
        save_dataframe(df_gps_for_zoning, path_gps_with_status)
    
    gps_with_zones = mapper.assign_zone_id(df_gps_for_zoning.copy(), lon_col='lon', lat_col='lat')
    logging.info(f"GPS with zones unique zones: {gps_with_zones['zone_id'].nunique()}")

    if gps_with_zones.empty:
        supply_features = pd.DataFrame(columns=['time_slot', 'zone_id', 'supply'])
    else:
        gps_with_zones['time_slot'] = gps_with_zones['device_time'].dt.floor(time_res)
        supply_features = gps_with_zones[gps_with_zones['status'] == 0].groupby(
            ['time_slot', 'zone_id'], observed=False
        )['vehicle_id'].nunique().reset_index(name='supply')
    logging.info(f"Supply features unique zones: {supply_features['zone_id'].nunique()}")

    # --- Part 3: Create and merge into a full spatio-temporal grid ---
    logging.info("Creating full spatio-temporal grid and merging base features...")
    
    # Create a list of series to concatenate for time slots and zone IDs
    time_slot_series = [df_orders_final['time_slot']]
    if not gps_with_zones.empty:
        time_slot_series.append(gps_with_zones['time_slot'])
    
    zone_id_series = [df_orders_final['origin_zone_id']]
    if not gps_with_zones.empty:
        zone_id_series.append(gps_with_zones['zone_id'])

    all_time_slots = pd.concat(time_slot_series).dropna().unique()
    all_zone_ids_from_data = pd.concat(zone_id_series).dropna().unique()
    logging.info(f"All unique zone IDs from orders and GPS data: {len(all_zone_ids_from_data)}")

    # Load all possible zone IDs from the GeoJSON file to ensure no zones are missed
    import geopandas as gpd
    geojson_path = config['spatio_temporal']['gridding']['geojson']['file_path']
    zone_id_property = config['spatio_temporal']['gridding']['geojson']['zone_id_property']
    all_possible_zones_gdf = gpd.read_file(geojson_path)
    all_possible_zone_ids = all_possible_zones_gdf[zone_id_property].astype(str).unique()
    logging.info(f"All possible zone IDs from GeoJSON: {len(all_possible_zone_ids)}")

    # Use all_possible_zone_ids for the full grid to ensure all zones are present
    full_grid = pd.MultiIndex.from_product(
        [all_time_slots, all_possible_zone_ids], 
        names=['time_slot', 'zone_id']
    ).to_frame(index=False)
    logging.info(f"Full grid created with {full_grid['zone_id'].nunique()} unique zones.")

    if len(all_time_slots) == 0 or len(all_possible_zone_ids) == 0:
        logging.warning("No time slots or zone IDs found. Returning empty DataFrame.")
        return pd.DataFrame()

    final_df = pd.merge(full_grid, demand_income_agg, on=['time_slot', 'zone_id'], how='left')
    final_df = pd.merge(final_df, supply_features, on=['time_slot', 'zone_id'], how='left')
    logging.info(f"Final DataFrame after merging demand/income/supply: {final_df['zone_id'].nunique()} unique zones.")

    # --- Part 4: Create target variables and add engineered features ---
    logging.info("Creating target variables and adding engineered features...")
    
    # Fill NaNs for base features and create target columns
    final_df[['trip_count', 'average_income', 'supply', 'total_income']] = final_df[['trip_count', 'average_income', 'supply', 'total_income']].fillna(0)
    final_df['has_orders'] = (final_df['trip_count'] > 0).astype(int)
    
    # Create 'demand' as a copy of 'trip_count' for compatibility with downstream feature functions
    final_df['demand'] = final_df['trip_count']

    # Add other features
    final_df = add_static_zone_features(final_df, config)
    final_df = add_time_features(final_df, 'time_slot', config)
    final_df = add_weather_features(final_df, config)
    final_df = add_advanced_interaction_features(final_df, config)
    
    final_df = final_df.sort_values(by=['zone_id', 'time_slot']).reset_index(drop=True)
    
    final_df = add_lagged_features(final_df, config)
    final_df = add_rolling_window_features(final_df, config)
    
    # --- Part 5: Final cleanup ---
    logging.info("Calculating final derived features and cleaning up...")
    final_df['demand_supply_gap'] = final_df['demand'] - final_df['supply']
    
    # Fill NaNs created by lagging/rolling at the start of the series
    cols_to_fill_na = [col for col in final_df.columns if col not in ['time_slot', 'zone_id'] and pd.api.types.is_numeric_dtype(final_df[col])]
    final_df[cols_to_fill_na] = final_df[cols_to_fill_na].fillna(0)
            
    logging.info(f"Final feature DataFrame created with {len(final_df)} rows and columns: {final_df.columns.tolist()}")
    return final_df