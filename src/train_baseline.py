# src/train_baseline.py
"""
Main training and evaluation pipeline for all baseline models.
This script is structured to be consistent with train_dhk.py.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Rich and Sklearn for better metrics and presentation
from rich.table import Table
from rich.console import Console
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F

# Add project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.dhk_dataloader import DHKIDataset
from baselines.statistical_models import HistoricalAverage
from baselines.deep_learning_baselines import LSTM_Baseline, RNN_Baseline, LSTM_GCN_Baseline, Transformer_Baseline
from baselines.st_models import STGCN_Baseline, DCRNN_Baseline, LSTM_GAT_Baseline
from utils.common import setup_logging, get_git_revision_hash, flatten_config

logger = logging.getLogger(__name__)

# --- Global variable for epoch number in progress bar ---
global epoch_num

def get_model(baseline_model: str, config: Dict, num_zones: int, num_ts_features: int, direct_income: bool) -> Any:
    """Factory function to get the baseline model instance."""
    model_config = config.get('model', {})
    if baseline_model == 'historical_average':
        return {target: {zone: HistoricalAverage() for zone in range(num_zones)} for target in ['has_orders', 'trip_count', 'avg_income']}
    elif baseline_model == 'random_forest':
        return {
            'has_orders': RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10, n_estimators=50),
            'trip_count': RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10, n_estimators=50),
            'avg_income': RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10, n_estimators=50)
        }
    elif baseline_model == 'lstm':
        return LSTM_Baseline(num_ts_features, model_config.get('embedding_dim', 64), direct_income=direct_income)
    elif baseline_model == 'rnn':
        return RNN_Baseline(num_ts_features, model_config.get('embedding_dim', 64), direct_income=direct_income)
    elif baseline_model == 'lstm_gcn':
        return LSTM_GCN_Baseline(num_ts_features, model_config.get('embedding_dim', 64), direct_income=direct_income)
    elif baseline_model == 'transformer':
        return Transformer_Baseline(
            num_input_features=num_ts_features,
            hidden_dim=model_config.get('embedding_dim', 64),
            num_heads=model_config.get('num_heads', 4),
            num_layers=model_config.get('num_transformer_layers', 2),
            max_seq_len=config['features']['historical_look_back_steps'],
            num_zones=num_zones,
            direct_income=direct_income
        )
    elif baseline_model == 'stgcn':
        return STGCN_Baseline(num_zones, num_ts_features, config['features']['historical_look_back_steps'], 1, direct_income=direct_income)
    elif baseline_model == 'dcrnn':
        return DCRNN_Baseline(num_zones, num_ts_features, model_config.get('embedding_dim', 64), 2, 1, direct_income=direct_income)
    elif baseline_model == 'lstm_gat':
        return LSTM_GAT_Baseline(num_ts_features, model_config.get('embedding_dim', 64), direct_income=direct_income)
    else:
        raise ValueError(f"Unknown baseline model: {baseline_model}")

def run_one_epoch(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    baseline_model: str,
    direct_income: bool,
    optimizer: torch.optim.Optimizer = None,
    epoch_num: int = 0, # Add epoch_num as an optional argument
) -> Tuple[float, List[Dict], List[Dict]]:
    """Runs a single epoch of training or evaluation for baseline models."""
    is_training = optimizer is not None
    if isinstance(model, torch.nn.Module):
        model.train(is_training)
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    desc = "Train" if is_training else "Eval"
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [{desc}]", leave=False)

    for batch in progress_bar:
        dynamic_features = batch["dynamic_features"].to(device)
        targets = batch['target'].to(device)
        edge_index = dataloader.dataset.edge_index.to(device) if hasattr(dataloader.dataset, 'edge_index') else None
        adj_matrix = batch['adjacency_matrix'].to(device) if 'adjacency_matrix' in batch else None

        if targets.shape[1] == 3 and targets.shape[2] != 3:
            targets = targets.permute(0, 2, 1)

        outputs = {}
        loss = 0.0 # Initialize loss for all cases

        with torch.set_grad_enabled(is_training):
            # Dispatch to the correct model with the correct graph representation
            if baseline_model in ['stgcn', 'lstm_gcn', 'lstm_gat']:
                if edge_index is None:
                    raise ValueError(f"Model {baseline_model} requires edge_index, but it is not available in the dataset.")
                model_outputs = model(dynamic_features, edge_index)
            elif baseline_model == 'dcrnn':
                if adj_matrix is None:
                    raise ValueError(f"Model {baseline_model} requires an adjacency_matrix, but it is not available in the dataset.")
                model_outputs = model(dynamic_features, adj_matrix)
            elif baseline_model in ['lstm', 'rnn', 'transformer']:
                model_outputs = model(dynamic_features)
            elif baseline_model == 'historical_average':
                # For HA, predictions are made per zone, per time_slot
                pred_has_orders_list, pred_count_list, pred_income_list = [], [], []
                for zone_idx in range(dataloader.dataset.num_zones):
                    pred_has_orders, pred_count, pred_income = model['has_orders'][zone_idx].predict(batch, dataloader.dataset)
                    pred_has_orders_list.append(pred_has_orders)
                    pred_count_list.append(pred_count)
                    pred_income_list.append(pred_income)

                pred_has_orders = torch.stack(pred_has_orders_list, dim=1).to(device)
                pred_count = torch.stack(pred_count_list, dim=1).to(device)
                pred_income = torch.stack(pred_income_list, dim=1).to(device)

                model_outputs = (pred_count, pred_income) # HA doesn't directly predict has_orders_logits
                outputs = {
                    'has_orders_logits': pred_has_orders, # Use raw prediction for has_orders
                    'log_trip_count_pred': pred_count,
                    'log_avg_income_pred': pred_income,
                }

            elif baseline_model == 'random_forest':
                df_batch = pd.DataFrame({
                    'time_slot': pd.to_datetime(np.repeat(batch['time_slot'].cpu().numpy(), dataloader.dataset.num_zones), unit='s'),
                    'zone_idx': np.tile(np.arange(dataloader.dataset.num_zones), len(batch['time_slot']))
                })
                df_batch['dayofweek'] = df_batch['time_slot'].dt.dayofweek
                df_batch['hour'] = df_batch['time_slot'].dt.hour
                features = ['dayofweek', 'hour', 'zone_idx']
                X_batch = df_batch[features]

                pred_has_orders = torch.from_numpy(model['has_orders'].predict(X_batch)).reshape(targets.shape[0], targets.shape[1]).to(device)
                pred_count = torch.from_numpy(model['trip_count'].predict(X_batch)).reshape(targets.shape[0], targets.shape[1]).to(device)
                pred_income = torch.from_numpy(model['avg_income'].predict(X_batch)).reshape(targets.shape[0], targets.shape[1]).to(device)

                model_outputs = (pred_count, pred_income)
                outputs = {
                    'has_orders_logits': pred_has_orders,
                    'log_trip_count_pred': pred_count,
                    'log_avg_income_pred': pred_income,
                }

            # Process outputs and calculate loss for deep learning models
            if isinstance(model, torch.nn.Module):
                if direct_income:
                    pred_total_income = model_outputs
                    loss = F.mse_loss(pred_total_income, batch['total_income'].to(device))
                    outputs = {'pred_total_income': pred_total_income}
                else:
                    pred_count, pred_income = model_outputs
                    has_orders_logits = pred_count # A common proxy
                    loss = (
                        F.binary_cross_entropy_with_logits(has_orders_logits, targets[:, :, 0]) + 
                        F.mse_loss(pred_count, targets[:, :, 1]) + 
                        F.mse_loss(pred_income, targets[:, :, 2])
                    )
                    outputs = {
                        'has_orders_logits': has_orders_logits,
                        'log_trip_count_pred': pred_count,
                        'log_avg_income_pred': pred_income
                    }
            else: # For non-deep learning models, calculate loss here
                # For HA and RF, outputs are already in 'outputs' dict
                if direct_income:
                    # Need to calculate pred_total_income from outputs for non-DL models
                    pred_total_income_calc = torch.sigmoid(outputs['has_orders_logits']) * torch.expm1(outputs['log_trip_count_pred']) * torch.expm1(outputs['log_avg_income_pred'])
                    loss = F.mse_loss(pred_total_income_calc, batch['total_income'].to(device))
                else:
                    loss = (
                        F.binary_cross_entropy_with_logits(outputs['has_orders_logits'], targets[:, :, 0]) + 
                        F.mse_loss(outputs['log_trip_count_pred'], targets[:, :, 1]) + 
                        F.mse_loss(outputs['log_avg_income_pred'], targets[:, :, 2])
                    )

        if is_training and isinstance(model, torch.nn.Module):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix_str(f"Loss: {loss.item():.4f}")

        # Detach all tensors before appending to the list
        if direct_income:
            detached_outputs = {k: v.detach().cpu() for k, v in outputs.items()}
        else:
            outputs['pred_total_income'] = torch.sigmoid(outputs['has_orders_logits']) * torch.expm1(outputs['log_trip_count_pred']) * torch.expm1(outputs['log_avg_income_pred'])
            detached_outputs = {k: v.detach().cpu() for k, v in outputs.items()}
        all_outputs.append(detached_outputs)

        all_targets.append({
            'has_orders': targets[:, :, 0].detach().cpu(),
            'log_trip_count': targets[:, :, 1].detach().cpu(),
            'log_avg_income': targets[:, :, 2].detach().cpu(),
            'total_income': batch['total_income'].detach().cpu()
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_outputs, all_targets

def evaluate_and_get_metrics(all_outputs: List[Dict], all_targets: List[Dict], direct_income: bool) -> Dict[str, float]:
    """Calculates all classification and regression metrics."""
    metrics = {}
    
    if not all_outputs:
        return {}
    if direct_income:
        pred_total_income = torch.cat([o['pred_total_income'] for o in all_outputs], dim=0).flatten()
    else:
        logits = torch.cat([o['has_orders_logits'] for o in all_outputs], dim=0).flatten()
        pred_count = torch.cat([o['log_trip_count_pred'] for o in all_outputs], dim=0).flatten()
        pred_income = torch.cat([o['log_avg_income_pred'] for o in all_outputs], dim=0).flatten()
        pred_total_income = torch.sigmoid(logits) * torch.expm1(pred_count) * torch.expm1(pred_income)

    true_total_income = torch.cat([t['total_income'] for t in all_targets], dim=0).flatten()

    metrics['MSE_total_income'] = F.mse_loss(pred_total_income, true_total_income).item()
    metrics['RMSE_total_income'] = torch.sqrt(torch.tensor(metrics['MSE_total_income'])).item()
    metrics['MAE_total_income'] = F.l1_loss(pred_total_income, true_total_income).item()

    return metrics

def main(config_path: str, baseline_model: str, time_interval_minutes: int, enable_time_window_filter: bool, direct_income_prediction: bool):
    """Main training and evaluation loop for baseline models."""
    global epoch_num
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if time_interval_minutes:
        config['spatio_temporal']['time_interval_minutes'] = time_interval_minutes
    if enable_time_window_filter:
        config['data_filtering']['time_window_filter']['enable'] = True

    # config['training']['epochs'] = 1
    # logger.info("Set epochs to 1 for local testing.")

    setup_logging(config)
    logger.info(f"Running baseline: {baseline_model}")
    logger.info(f"Git revision hash: {get_git_revision_hash()}")
    logger.info(f"Loaded configuration from {config_path}")

    device = torch.device(config['training']['device'])
    print(device)

    logger.info("Loading datasets...")
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = DHKIDataset(config=config, dataset_type=split)
        dataloaders[split] = DataLoader(
            dataset, batch_size=config['training']['batch_size'], shuffle=(split == 'train'),
            num_workers=config.get('system', {}).get('num_workers', 0),
            pin_memory=True, persistent_workers=False
        )
    logger.info("Datasets loaded successfully.")

    train_dataset = dataloaders['train'].dataset
    model = get_model(baseline_model, config, train_dataset.num_zones, train_dataset.num_input_features, direct_income_prediction)
    
    if isinstance(model, torch.nn.Module):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        best_val_loss = float('inf')
        best_model_path = Path(config['training']['checkpoint_path']) / f"best_model_{baseline_model}.pth"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training loop for deep learning model...")
        for epoch in range(config['training']['epochs']):
            train_loss, _, _ = run_one_epoch(model, dataloaders['train'], device, baseline_model, direct_income_prediction, optimizer, epoch_num=epoch)
            with torch.no_grad():
                val_loss, _, _ = run_one_epoch(model, dataloaders['val'], device, baseline_model, direct_income_prediction, epoch_num=epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"New best model saved to {best_model_path} (Val Loss: {val_loss:.4f})")
        logger.info("Training complete. Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.info(f"Fitting {baseline_model} model...")
        all_train_data = []
        for batch in tqdm(dataloaders['train'], desc=f"Aggregating data for {baseline_model}"):
            targets_np = batch['target'].numpy()
            targets_np[:, :, 1] = np.expm1(targets_np[:, :, 1])
            targets_np[:, :, 2] = np.expm1(targets_np[:, :, 2])

            for i, time_slot in enumerate(batch['time_slot']):
                for zone_idx in range(train_dataset.num_zones):
                    all_train_data.append({
                        'time_slot': pd.to_datetime(time_slot.item(), unit='s'),
                        'zone_idx': zone_idx,
                        'has_orders': targets_np[i, zone_idx, 0],
                        'trip_count': targets_np[i, zone_idx, 1],
                        'avg_income': targets_np[i, zone_idx, 2],
                    })
        df_train = pd.DataFrame(all_train_data)

        if baseline_model == 'historical_average':
            for target in ['has_orders', 'trip_count', 'avg_income']:
                for zone_idx in range(train_dataset.num_zones):
                    zone_df = df_train[df_train['zone_idx'] == zone_idx]
                    model[target][zone_idx].fit(zone_df, 'time_slot', target)
        elif baseline_model == 'random_forest':
            df_train['dayofweek'] = df_train['time_slot'].dt.dayofweek
            df_train['hour'] = df_train['time_slot'].dt.hour
            features = ['dayofweek', 'hour', 'zone_idx']
            X_train = df_train[features]
            for target in ['has_orders', 'trip_count', 'avg_income']:
                y_train = df_train[target]
                model[target].fit(X_train, y_train)
        logger.info(f"Fitting complete.")

    final_metrics = {}
    train_metrics = evaluate_and_get_metrics([], [], direct_income_prediction)
    val_metrics = evaluate_and_get_metrics([], [], direct_income_prediction)

    if isinstance(model, torch.nn.Module):
        # Evaluation for deep learning models
        logger.info("Evaluating on train set...")
        _, train_outputs, train_targets = run_one_epoch(model, dataloaders['train'], device, baseline_model, direct_income_prediction)
        train_metrics = evaluate_and_get_metrics(train_outputs, train_targets, direct_income_prediction)
        logger.info("Evaluating on val set...")
        _, val_outputs, val_targets = run_one_epoch(model, dataloaders['val'], device, baseline_model, direct_income_prediction)
        val_metrics = evaluate_and_get_metrics(val_outputs, val_targets, direct_income_prediction)
    
    final_metrics['train'] = train_metrics
    final_metrics['val'] = val_metrics

    # Final evaluation on test set and save raw outputs
    logger.info(f"Evaluating on test set...")
    _, test_outputs, test_targets = run_one_epoch(model, dataloaders['test'], device, baseline_model, direct_income_prediction)
    final_metrics['test'] = evaluate_and_get_metrics(test_outputs, test_targets, direct_income_prediction)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Train")
    table.add_column("Validation")
    table.add_column("Test")

    metric_keys = list(final_metrics.get('train', {}).keys())
    for key in metric_keys:
        table.add_row(
            key,
            f"{final_metrics.get('train', {}).get(key, 0):.4f}",
            f"{final_metrics.get('val', {}).get(key, 0):.4f}",
            f"{final_metrics.get('test', {}).get(key, 0):.4f}"
        )
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(f"results/runs_baseline/{config['spatio_temporal']['time_interval_minutes']}min")
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = 'direct' if direct_income_prediction else ''
    results_filename = results_dir / f"run_{baseline_model}_{suffix_str}_{timestamp}.json"

    # Extract final predictions and targets for saving
    test_predictions_tensor = torch.cat([o['pred_total_income'] for o in test_outputs], dim=0).flatten()
    test_targets_tensor = torch.cat([t['total_income'] for t in test_targets], dim=0).flatten()

    run_data = {
        "timestamp": timestamp,
        "baseline_model": baseline_model,
        "git_hash": get_git_revision_hash(),
        "config": config,
        "metrics": final_metrics,
        "test_predictions": test_predictions_tensor.tolist(),
        "test_targets": test_targets_tensor.tolist()
    }

    with open(results_filename, 'w') as f:
        json.dump(run_data, f, indent=4)
    
    logger.info(f"Final results and configuration saved to {results_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate baseline models')
    parser.add_argument('--config', type=str, default='config/main_config.yaml', help='Path to config file')
    parser.add_argument('--baseline_model', required=True, type=str, 
                        choices=['historical_average', 'random_forest', 'lstm', 'rnn', 'lstm_gcn', 'transformer', 'stgcn', 'dcrnn', 'lstm_gat'],
                        help='The baseline model to run.')
    parser.add_argument('--time_interval_minutes', type=int, default=None, help='Override time interval in minutes.')
    parser.add_argument('--enable_time_window_filter', action='store_true', help='Enable the time window filter.')
    parser.add_argument('--direct_income_prediction', action='store_true', help='Directly predict total income instead of decoupled targets.')
    args = parser.parse_args()
    main(args.config, args.baseline_model, args.time_interval_minutes, args.enable_time_window_filter, args.direct_income_prediction)