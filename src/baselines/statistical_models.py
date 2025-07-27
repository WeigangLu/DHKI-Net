#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script implements classical statistical models for time series forecasting baselines.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from dataset.dhk_dataloader import DHKIDataset
import torch

logger = logging.getLogger(__name__)

class HistoricalAverage:
    """
    A baseline model that predicts the value for a given time slot based on the
    historical average of all previous occurrences of that same time slot.
    (e.g., the prediction for Tuesday at 10:00 AM is the average of all previous
    Tuesdays at 10:00 AM).
    """
    def __init__(self):
        self.averages = {}
        self.global_average = 0

    def fit(self, history: pd.DataFrame, time_col: str, target_col: str):
        """
        Calculates the historical averages from the training data.

        Args:
            history (pd.DataFrame): A DataFrame with a datetime column and a target column.
            time_col (str): The name of the datetime column.
            target_col (str): The name of the column to be averaged.
        """
        logger.info("Fitting HistoricalAverage model...")
        df = history.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df['dayofweek'] = df[time_col].dt.dayofweek
        df['hour'] = df[time_col].dt.hour
        
        self.averages = df.groupby(['dayofweek', 'hour'])[target_col].mean().to_dict()
        self.global_average = df[target_col].mean()
        logger.info(f"Model fitted. Found {len(self.averages)} historical averages.")

    def predict(self, batch: dict, dataset: DHKIDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Makes predictions for a given batch of data.
        """
        time_slots = pd.to_datetime(batch['time_slot'].numpy(), unit='s')
        batch_size = len(time_slots)

        pred_orders = np.zeros((batch_size))
        pred_count = np.zeros((batch_size))
        pred_income = np.zeros((batch_size))

        for i, ts in enumerate(time_slots):
            key = (ts.dayofweek, ts.hour)
            pred_orders[i] = self.averages.get(key, self.global_average)
            pred_count[i] = self.averages.get(key, self.global_average)
            pred_income[i] = self.averages.get(key, self.global_average)

        return torch.from_numpy(pred_orders).float(), torch.from_numpy(pred_count).float(), torch.from_numpy(pred_income).float()

class SARIMA_model:
    """
    A wrapper for the statsmodels SARIMAX model to provide a simple fit/predict interface.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        """
        Initializes the SARIMAX model wrapper.

        Args:
            order (tuple): The (p, d, q) order of the model.
            seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.
                                    The seasonal period 's' is crucial (e.g., 24 for hourly data with daily seasonality).
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, history: pd.Series):
        """
        Fits the SARIMAX model to the historical data.

        Args:
            history (pd.Series): A Series of historical data, indexed by time.
        """
        logger.info(f"Fitting SARIMAX model with order={self.order} and seasonal_order={self.seasonal_order}...")
        try:
            model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            self.model_fit = model.fit(disp=False)
            logger.info("SARIMAX model fitted successfully.")
        except Exception as e:
            logger.error(f"Failed to fit SARIMAX model: {e}")
            self.model_fit = None

    def predict(self, horizon_steps: int) -> pd.Series:
        """
        Makes predictions for a given number of steps into the future.

        Args:
            horizon_steps (int): The number of steps to forecast.

        Returns:
            pd.Series: A Series containing the forecast.
        """
        if self.model_fit is None:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")
        
        logger.info(f"Predicting {horizon_steps} steps into the future with SARIMAX...")
        return self.model_fit.forecast(steps=horizon_steps)
