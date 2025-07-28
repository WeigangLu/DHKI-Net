# src/utils/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import pandas as pd


def calculate_demand_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates regression metrics for the demand prediction task.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: A dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Mean Absolute Percentage Error (MAPE), avoiding division by zero
    y_true_nonzero = y_true[y_true > 0]
    y_pred_nonzero = y_pred[y_true > 0]
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100 if len(y_true_nonzero) > 0 else -1.0

    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}  # Return WAPE instead of MAPE

def compute_metrics(all_od_logits, all_od_targets, all_utility_gt):
    """
    Compute Top-k accuracy and Good Destination Recall metrics.
    """
    metrics = {}
    k_values = [1, 3, 5, 10]
    valid_mask = all_od_targets != -1
    if not torch.any(valid_mask):
        return {}
    all_od_logits = all_od_logits[valid_mask]
    all_od_targets = all_od_targets[valid_mask]
    all_utility_gt = all_utility_gt[valid_mask]
    _, top_k_preds = torch.topk(all_od_logits, k=max(k_values), dim=-1)
    for k in k_values:
        correct_k = top_k_preds[:, :k] == all_od_targets.unsqueeze(-1)
        metrics[f'Top-{k}_Accuracy'] = torch.any(correct_k, dim=1).float().mean().item()
    good_destination_mask = all_utility_gt == 1
    if torch.any(good_destination_mask):
        for k in k_values:
            good_targets = all_od_targets[good_destination_mask]
            good_preds_k = top_k_preds[:, :k][good_destination_mask]
            correct_good_k = good_preds_k == good_targets.unsqueeze(-1)
            metrics[f'Good_Destination_Recall@{k}'] = torch.any(correct_good_k, dim=1).float().mean().item()
    else:
        for k in k_values:
            metrics[f'Good_Destination_Recall@{k}'] = -1
    return metrics