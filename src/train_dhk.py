# src/train_dhk.py
"""
Main training and evaluation pipeline for the DHKI-Net.
"""

import argparse
import torch
import numpy as np
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
import torch.nn as nn
import torch.nn.functional as F


# Add project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.dhk_dataloader import DHKIDataset
from models.model_factory import create_model
from utils.common import setup_logging, get_git_revision_hash, flatten_config
from utils.loss import DHKILoss

logger = logging.getLogger(__name__)

def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: DHKILoss,
    device: torch.device,
    has_contrastive: bool,
    direct_income_prediction: bool, # New argument
    optimizer: torch.optim.Optimizer = None,
    epoch_num: int = 0,
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    """
    Runs a single epoch of training or evaluation.
    """
    is_training = optimizer is not None
    model.train(is_training)
    total_losses = {k: 0.0 for k in ['total_loss', 'BCE_Loss', 'MSE_Count', 'Huber_Income', 'Contrastive_Loss', 'Direct_Income_MSE']}

    all_outputs = []
    all_targets = []

    desc = "Train" if is_training else "Eval"
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [{desc}]", leave=False)

    for i, batch in enumerate(progress_bar):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        with torch.set_grad_enabled(is_training):
            outputs = model(
                dynamic_features=batch["dynamic_features"],
                zone_indices=batch["zone_indices"],
                static_features_all_zones=batch["static_features"],
                haversine_distance_matrix=batch["haversine_distance_matrix"],
                static_features_anchor=batch.get('static_features_anchor'),
                static_features_positive=batch.get('static_features_positive'),
                static_features_negatives=batch.get('static_features_negatives'),
                direct_income_prediction=direct_income_prediction
            )
            
            if direct_income_prediction:
                loss = F.mse_loss(outputs['pred_total_income'], batch['total_income'])
                loss_dict = {'total_loss': loss, 'Direct_Income_MSE': loss}
            else:
                loss, loss_dict = loss_fn(outputs, batch, has_contrastive=has_contrastive)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in loss_dict.items():
            if k in total_losses:
                total_losses[k] += v.item()
        
        progress_bar.set_postfix_str(f"Loss: {loss.item():.4f}")

        if not direct_income_prediction:
            outputs['pred_total_income'] = torch.sigmoid(outputs['has_orders_logits']) * torch.expm1(outputs['log_trip_count_pred']) * torch.expm1(outputs['log_avg_income_pred'])
        
        all_outputs.append({k: v.detach().cpu() for k, v in outputs.items()})
        all_targets.append({
            'has_orders': batch['target'][:, :, 0].detach().cpu(),
            'log_trip_count': batch['target'][:, :, 1].detach().cpu(),
            'log_avg_income': batch['target'][:, :, 2].detach().cpu(),
            'total_income': batch['total_income'].detach().cpu()
        })

    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items() if v > 0}
    return avg_losses, all_outputs, all_targets

def evaluate_and_get_metrics(
    all_outputs: List[Dict],
    all_targets: List[Dict],
    direct_income_prediction: bool, # New argument
    use_oracle_mask: bool = False,
    use_oracle_trip_count: bool = False,
    use_oracle_avg_income: bool = False
) -> Dict[str, float]:
    """Calculates all classification and regression metrics."""
    metrics = {}
    
    if not all_outputs:
        return {}

    if direct_income_prediction:
        pred_total_income = torch.cat([o['pred_total_income'] for o in all_outputs], dim=0).flatten()
    else:
        logits = torch.cat([o['has_orders_logits'] for o in all_outputs], dim=0).flatten()
        y_true_class = torch.cat([t['has_orders'] for t in all_targets], dim=0).flatten()
        pred_class = (torch.sigmoid(logits) > 0.5).int()
        metrics['Accuracy'] = accuracy_score(y_true_class, pred_class)
        metrics['Precision'] = precision_score(y_true_class, pred_class, zero_division=0)
        metrics['Recall'] = recall_score(y_true_class, pred_class, zero_division=0)
        metrics['F1-Score'] = f1_score(y_true_class, pred_class, zero_division=0)

        pred_count = torch.cat([o['log_trip_count_pred'] for o in all_outputs], dim=0).flatten()
        y_true_count = torch.cat([t['log_trip_count'] for t in all_targets], dim=0).flatten()

        pred_income = torch.cat([o['log_avg_income_pred'] for o in all_outputs], dim=0).flatten()
        y_true_income = torch.cat([t['log_avg_income'] for t in all_targets], dim=0).flatten()

        if use_oracle_mask:
            pred_prob_has_orders = y_true_class.float()
        else:
            pred_prob_has_orders = torch.sigmoid(logits)

        if use_oracle_trip_count:
            pred_trip_count = torch.expm1(y_true_count)
        else:
            pred_trip_count = torch.expm1(pred_count)

        if use_oracle_avg_income:
            pred_avg_income = torch.expm1(y_true_income)
        else:
            pred_avg_income = torch.expm1(pred_income)

        pred_total_income = pred_prob_has_orders * torch.clamp(pred_trip_count, min=0) * torch.clamp(pred_avg_income, min=0)

    true_total_income = torch.cat([t['total_income'] for t in all_targets], dim=0).flatten()

    metrics['MSE_total_income'] = F.mse_loss(pred_total_income, true_total_income).item()
    metrics['RMSE_total_income'] = torch.sqrt(torch.tensor(metrics['MSE_total_income'])).item()
    metrics['MAE_total_income'] = F.l1_loss(pred_total_income, true_total_income).item()

    return metrics

def main(args: argparse.Namespace):
    """Main training and evaluation loop."""
    global epoch_num
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger.info(f"Git revision hash: {get_git_revision_hash()}")
    logger.info(f"Loaded configuration from {args.config}")
    if args.disable_contrastive_learning:
        logger.info("Contrastive learning is DISABLED for this run.")

    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'device', 'disable_contrastive_learning', 'use_oracle_mask', 'use_oracle_trip_count', 'use_oracle_avg_income', 'enable_time_window_filter']:
            # Find the correct section in the config to override
            if key in config['training']:
                config['training'][key] = value
            elif key in config['model']:
                config['model'][key] = value
            elif key in config['data']:
                config['data'][key] = value
            elif key in config['spatio_temporal']:
                 config['spatio_temporal'][key] = value
    
    if args.enable_time_window_filter:
        config['data_filtering']['time_window_filter']['enable'] = True

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device(args.device if args.device else config['training']['device'])
    logger.info(f"Using device: {device}")
    has_contrastive = not args.disable_contrastive_learning

    logger.info("Loading datasets...")
    dataloaders = {}
    num_input_features, num_zones = 0, 0
    for split in ['train', 'val', 'test']:
        dataset = DHKIDataset(config=config, dataset_type=split)
        if split == 'train':
            num_input_features = dataset.num_input_features
            num_zones = dataset.num_zones
        dataloaders[split] = DataLoader(
            dataset, batch_size=config['training']['batch_size'], shuffle=(split == 'train'),
            num_workers=config.get('system', {}).get('num_workers', 0), pin_memory=True,
            persistent_workers=(config.get('system', {}).get('num_workers', 0) > 0)
        )
    logger.info("Datasets loaded successfully.")

    model = create_model(
        config=config, num_zones=num_zones, num_ts_features=num_input_features,
        disable_spatial_bias=args.disable_spatial_bias,
        value_head_type=args.value_head_type
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay']
    )
    loss_fn = DHKILoss(config)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config['training'].get('early_stopping_patience', 10)
    min_delta = config['training'].get('min_delta', 0.0001)
    best_model_path = Path(config['training']['checkpoint_path']) / f"best_model_{run_timestamp}.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop...")
    for epoch in range(config['training']['epochs']):
        epoch_num = epoch
        train_losses, train_outputs, train_targets = run_one_epoch(model, dataloaders['train'], loss_fn, device, has_contrastive, args.direct_income_prediction, optimizer)
        train_metrics = evaluate_and_get_metrics(train_outputs, train_targets, args.direct_income_prediction)

        with torch.no_grad():
            val_losses, val_outputs, val_targets = run_one_epoch(model, dataloaders['val'], loss_fn, device, has_contrastive, args.direct_income_prediction)
            val_metrics = evaluate_and_get_metrics(val_outputs, val_targets, args.direct_income_prediction)

        val_loss_key = 'Direct_Income_MSE' if args.direct_income_prediction else 'RMSE_total_income'
        current_val_loss = val_losses.get(val_loss_key, val_metrics.get(val_loss_key, float('inf')))

        logger.info(f"Epoch {epoch} | Train Loss: {train_losses['total_loss']:.4f} | Val Loss: {val_losses['total_loss']:.4f} | Val RMSE: {val_metrics.get('RMSE_total_income', 0):.4f}")

        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete. Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    final_metrics = {}
    final_metrics['train'] = train_metrics
    final_metrics['val'] = val_metrics

    # Final evaluation on test set and save raw outputs
    logger.info("Evaluating on test set...")
    with torch.no_grad():
        _, test_outputs, test_targets = run_one_epoch(model, dataloaders['test'], loss_fn, device, has_contrastive, args.direct_income_prediction)
        final_metrics['test'] = evaluate_and_get_metrics(test_outputs, test_targets, args.direct_income_prediction, args.use_oracle_mask, args.use_oracle_trip_count, args.use_oracle_avg_income)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    for split in ['train', 'val', 'test']: table.add_column(split.capitalize())
    
    for key in final_metrics['train']:
        table.add_row(key, *[f"{final_metrics[s][key]:.4f}" for s in ['train', 'val', 'test']])
    console.print(table)

    # Determine results directory
    ablation_name = None
    if args.direct_income_prediction:
        ablation_name = "direct_income"
    elif args.disable_contrastive_learning:
        ablation_name = "no_contrastive"
    elif args.disable_spatial_bias:
        ablation_name = "no_spatial_bias"
    elif args.value_head_type != 'attention':
        ablation_name = f"value_head_{args.value_head_type}"

    if ablation_name:
        results_dir = Path(f"results/ablations/{ablation_name}/{config['spatio_temporal']['time_interval_minutes']}min")
    else:
        results_dir = Path(f"results/runs/{config['spatio_temporal']['time_interval_minutes']}min")

    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"run_{run_timestamp}.json"
    config_save_path = best_model_path.parent / f"run_{run_timestamp}_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved final config to {config_save_path}")

    # Extract final predictions and targets for saving
    test_predictions_tensor = torch.cat([o['pred_total_income'] for o in test_outputs], dim=0).flatten()
    test_targets_tensor = torch.cat([t['total_income'] for t in test_targets], dim=0).flatten()

    run_data = {
        "timestamp": run_timestamp, 
        "git_hash": get_git_revision_hash(), 
        "config": config, 
        "metrics": final_metrics,
        "test_predictions": test_predictions_tensor.tolist(),
        "test_targets": test_targets_tensor.tolist()
    }
    with open(results_filename, 'w') as f:
        json.dump(run_data, f, indent=4)
    logger.info(f"Final results saved to {results_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DHKI-Net')
    # Core args
    parser.add_argument('--config', type=str, default='config/main_config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cpu, cuda:0)')
    # Ablation args
    parser.add_argument('--direct_income_prediction', action='store_true', help='Directly predict total income instead of decoupled targets.')
    parser.add_argument('--disable_contrastive_learning', action='store_true', help='Disable contrastive learning')
    parser.add_argument('--disable_spatial_bias', action='store_true', help='Disable relative position bias')
    parser.add_argument('--value_head_type', type=str, default='attention', choices=['attention', 'mlp'], help='Type of value prediction head')
    # Oracle flags
    parser.add_argument('--use_oracle_mask', action='store_true', help='Use ground truth for has_orders')
    parser.add_argument('--use_oracle_trip_count', action='store_true', help='Use ground truth for trip_count')
    parser.add_argument('--use_oracle_avg_income', action='store_true', help='Use ground truth for avg_income')
    # Hyperparameters
    parser.add_argument('--fusion_module_type', type=str, default=None, choices=['gated', 'additive', 'bilinear'])
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--d_learnable', type=int, default=None)
    parser.add_argument('--d_pos', type=int, default=None)
    parser.add_argument('--num_contrastive_negatives', type=int, default=None)
    parser.add_argument('--num_transformer_layers', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--bce_loss_weight', type=float, default=None)
    parser.add_argument('--count_loss_weight', type=float, default=None)
    parser.add_argument('--income_loss_weight', type=float, default=None)
    parser.add_argument('--contrastive_loss_weight', type=float, default=None)
    parser.add_argument('--contrastive_temperature', type=float, default=None)
    # Data args
    parser.add_argument('--time_interval_minutes', type=int, default=None)
    parser.add_argument('--enable_time_window_filter', action='store_true')

    args = parser.parse_args()
    main(args)