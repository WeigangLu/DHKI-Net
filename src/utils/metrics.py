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


# --- 2. 模块化的指标计算 ---
def calculate_classification_metrics(
    all_predicted_logits: torch.Tensor,
    all_od_target_indices: torch.Tensor,
    all_utility_class_targets: torch.Tensor,
    df_gt_source_eval: pd.DataFrame,
    zone_to_idx: dict
) -> dict:
    """在测试集上计算多分类任务的完整评估指标。"""
    # logger.info("Calculating detailed classification metrics...")
    metrics = {}
    
    valid_mask = all_od_target_indices != -1
    if not torch.any(valid_mask):
        return {'error': 'No valid targets for metric calculation.'}

    # 筛选有效数据
    preds_all_zones_logits = all_predicted_logits[valid_mask]
    target_indices = all_od_target_indices[valid_mask]
    actual_utility_classes = all_utility_class_targets[valid_mask]

    # --- 指标 1: 真实终点的效用等级预测准确率 ---
    predicted_logits_for_actual = preds_all_zones_logits[torch.arange(len(preds_all_zones_logits)), target_indices]
    predicted_classes = torch.argmax(predicted_logits_for_actual, dim=1)
    metrics['Utility_Class_Accuracy'] = (predicted_classes == actual_utility_classes).float().mean().item()

    # --- 指标 2: NDCG@K (Normalized Discounted Cumulative Gain) ---
    k_values = [5, 10, 20]
    
    # 步骤 A: 动态计算每个类别的增益 (Gain)
    num_classes = all_utility_class_targets.max().item() + 1  # 类别从0开始
    # num_classes = len(config['experiment']['ground_truth']['utility_class_bins']) - 1
    # Gain = (N - class_label), 类别0的增益最高
    gain_mapping = torch.tensor([float(num_classes - 1 - i) for i in range(num_classes)])
    
    # 步骤 B: 计算每个区域的历史平均真实增益
    historical_mean_class = df_gt_source_eval.groupby('destination_zone_id')['utility_class'].mean()
    true_gain_tensor = torch.full((len(zone_to_idx),), fill_value=0.0) # 未出现的区域增益为0
    for zone, mean_class in historical_mean_class.items():
        if zone in zone_to_idx:
            # 使用四舍五入后的类别来查找增益
            true_gain_tensor[zone_to_idx[zone]] = gain_mapping[int(round(mean_class))]
            
    # 步骤 C: 计算 DCG@K
    # 推荐策略：推荐模型认为属于最优类别(Class 0)概率最高的目的地
    probs_all_zones = torch.softmax(preds_all_zones_logits, dim=-1)
    prob_of_being_best_class = probs_all_zones[:, :, 0]
    _, top_k_indices = torch.topk(prob_of_being_best_class, k=max(k_values), dim=-1)
    
    # 获取推荐列表的真实增益
    # true_gains_of_recs 形状: (N_trips, k_max)
    true_gains_of_recs = true_gain_tensor[top_k_indices]
    
    # 计算折损
    discounts = 1 / torch.log2(torch.arange(max(k_values)) + 2.0)
    
    # 计算每个K值的DCG
    dcg_at_k = {}
    for k in k_values:
        dcg_at_k[k] = (true_gains_of_recs[:, :k] * discounts[:k]).sum(dim=-1)
        
    # 步骤 D: 计算 IDCG@K (理想DCG)
    # 理想排序是按真实增益从高到低排列
    ideal_gains, _ = torch.sort(true_gain_tensor, descending=True)
    idcg_at_k = {}
    for k in k_values:
        idcg_at_k[k] = (ideal_gains[:k] * discounts[:k]).sum()
        
    # 步骤 E: 计算 NDCG@K
    for k in k_values:
        # 避免除以零 (如果所有区域的真实增益都是0)
        if idcg_at_k[k] > 0:
            ndcg = (dcg_at_k[k] / idcg_at_k[k]).mean().item()
        else:
            ndcg = 0.0
        metrics[f'NDCG@{k}'] = ndcg

    return metrics