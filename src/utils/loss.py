import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class ContrastiveLoss(nn.Module):
    """
    Implements the InfoNCE (Noise-Contrastive Estimation) loss.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)
        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        l_pos = torch.bmm(anchor, positive.transpose(1, 2)).squeeze(-1)
        l_neg = torch.bmm(anchor, negatives.transpose(1, 2)).squeeze(1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

class DHKILoss(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.bce_weight = config['training']['bce_loss_weight']
        self.count_weight = config['training']['count_loss_weight']
        self.income_weight = config['training']['income_loss_weight']
        self.contrastive_weight = config['training']['contrastive_loss_weight']

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.5074))
        self.count_loss = nn.MSELoss(reduction='none')
        self.income_loss = nn.HuberLoss(reduction='none') # Use HuberLoss for robustness to outliers
        if self.contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss(temperature=config['training']['contrastive_temperature'])

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], has_contrastive: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the total loss for the DHKI-Net model using a masked approach
        for regression losses, as specified in the methodology.
        """
        # --- 1. Create the Binary Mask (m) ---
        # The mask is based on the ground truth `has_orders` label.
        # p in the paper corresponds to has_orders.
        mask = batch['target'][:, :, 0]

        # --- 2. Calculate Supervised Losses ---
        # Likelihood Loss (L_p) - Calculated over all samples
        bce_loss = self.bce_loss(outputs['has_orders_logits'], mask)

        # Regression Losses (L_c, L_v) - Calculated only for samples where orders exist
        # First, calculate per-sample losses
        per_sample_count_loss = self.count_loss(outputs['log_trip_count_pred'], batch['target'][:, :, 1])
        per_sample_income_loss = self.income_loss(outputs['log_avg_income_pred'], batch['target'][:, :, 2])

        # Apply the mask
        masked_count_loss = per_sample_count_loss * mask
        masked_income_loss = per_sample_income_loss * mask

        # Sum and normalize by the number of non-zero samples
        num_positive_samples = mask.sum()
        if num_positive_samples > 0:
            count_loss = masked_count_loss.sum() / num_positive_samples
            income_loss = masked_income_loss.sum() / num_positive_samples
        else:
            # Handle edge case where a batch has no positive samples
            count_loss = torch.tensor(0.0, device=mask.device)
            income_loss = torch.tensor(0.0, device=mask.device)

        # --- 3. Calculate Contrastive Loss (L_CL) ---
        if has_contrastive and self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(
                outputs['anchor_embedding'],
                outputs['positive_embedding'],
                outputs['negative_embeddings']
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=bce_loss.device)

        # --- 4. Combine All Losses into the Final Objective ---
        total_loss = (
            bce_loss + # alpha is implicitly 1
            self.count_weight * count_loss +       # beta in the paper
            self.income_weight * income_loss +     # gamma in the paper
            self.contrastive_weight * contrastive_loss
        )

        loss_components = {
            'total_loss': total_loss,
            'BCE_Loss': bce_loss,
            'MSE_Count': count_loss,
            'Huber_Income': income_loss,
            'Contrastive_Loss': contrastive_loss,
        }

        return total_loss, loss_components

    def info_nce_loss(self, anchor, positive, negatives):
        """InfoNCE loss for contrastive learning."""
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        # Positive similarity
        l_pos = torch.bmm(anchor.unsqueeze(1), positive.unsqueeze(2)).squeeze(-1)

        # Negative similarity
        l_neg = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.config['training']['contrastive_temperature']

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
