#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script contains functions for creating handcrafted features for classical
machine learning models like XGBoost.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_time_series_features(dataframe: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """
    Engineers a rich set of features from a time-series dataframe.

    This function is designed to be applied to the time series of a single zone.

    Args:
        dataframe (pd.DataFrame): The input dataframe with a datetime column and a target column.
        time_col (str): The name of the datetime column (e.g., 'time_slot').
        target_col (str): The name of the target column (e.g., 'demand').

    Returns:
        pd.DataFrame: A new dataframe with the original data and many new features.
    """
    logger.info(f"Creating time series features for target '{target_col}'...")
    df = dataframe.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    # Ensure a continuous time index for proper shifting and rolling
    df = df.set_index(time_col).sort_index()
    # Find the time interval from the data
    time_interval = df.index.to_series().diff().median()
    df = df.asfreq(time_interval)

    # --- 1. Time-based Features ---
    logger.info("  Creating time-based features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df.index.quarter

    # --- 2. Lag Features ---
    logger.info("  Creating lag features...")
    lag_hours = [1, 2, 3, 6, 12, 24]
    for h in lag_hours:
        df[f'{target_col}_lag_{h}h'] = df[target_col].shift(h)

    # --- 3. Rolling Window Features ---
    logger.info("  Creating rolling window features...")
    rolling_windows = [3, 24] # in hours
    for w in rolling_windows:
        rolling_obj = df[target_col].rolling(window=w, min_periods=1)
        df[f'{target_col}_roll_{w}h_mean'] = rolling_obj.mean()
        df[f'{target_col}_roll_{w}h_std'] = rolling_obj.std()
        df[f'{target_col}_roll_{w}h_min'] = rolling_obj.min()
        df[f'{target_col}_roll_{w}h_max'] = rolling_obj.max()

    # Fill NaNs created by lagging/rolling
    df.fillna(method='bfill', inplace=True)

    logger.info("Finished creating time series features.")
    return df.reset_index()
