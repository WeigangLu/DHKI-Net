# src/models/dhk_net.py
"""
This module implements the DHKI-Net model with the GNN component removed
and replaced by a sophisticated Relative Position-Aware Transformer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import logging

from models.components import RelativePositionBias, TransformerEncoderLayer, ClassificationHead, RegressionHead, AttentionHead
from models.contextual_stream import ContextualStream
from models.fusion_module import GatedFusion

logger = logging.getLogger(__name__)

class DHKINet(nn.Module):
    """
    DHKI-Net architecture with Relative Position-Aware Transformer in Observational Stream.
    """
    def __init__(self, config: Dict, num_zones: int, num_ts_features: int, 
                 observational_stream: nn.Module, contextual_stream: nn.Module, 
                 fusion_module: nn.Module, classification_head: nn.Module, 
                 trip_count_head: nn.Module, avg_income_head: nn.Module, 
                 disable_spatial_bias: bool = False):
        super().__init__()
        self.config = config
        self.num_zones = num_zones
        self.disable_spatial_bias = disable_spatial_bias
        model_dim = config['model']['embedding_dim']
        d_learnable = config['model']['d_learnable']
        self.num_heads = config['model']['num_heads']
        num_distance_buckets = config['model']['num_distance_buckets']

        # 1. Observational Stream Components
        self.phi_emb = nn.Embedding(num_zones, d_learnable)
        self.input_projection = nn.Linear(num_ts_features + d_learnable, model_dim)
        if self.disable_spatial_bias:
            self.positional_encoding = nn.Embedding(num_zones, model_dim)
        else:
            self.relative_position_bias = RelativePositionBias(num_distance_buckets)
        self.transformer_encoder_layers = observational_stream
        self.transformer_norm = nn.LayerNorm(model_dim)

        # 2. Contextual Stream
        self.contextual_stream = contextual_stream

        # 3. Hyper-Knowledge Infusion Module
        self.fusion_module = fusion_module

        # 4. Prediction Heads
        self.classification_head = classification_head
        self.trip_count_head = trip_count_head
        self.avg_income_head = avg_income_head
        # Add a new head for direct income prediction
        self.direct_income_head = RegressionHead(model_dim)

    def forward(self,
                dynamic_features: torch.Tensor,
                zone_indices: torch.Tensor,
                static_features_all_zones: torch.Tensor,
                haversine_distance_matrix: torch.Tensor,
                static_features_anchor: torch.Tensor = None,
                static_features_positive: torch.Tensor = None,
                static_features_negatives: torch.Tensor = None,
                direct_income_prediction: bool = False
               ) -> Dict[str, torch.Tensor]:

        # dynamic_features shape: (batch_size, seq_len, num_zones, num_ts_features)
        # zone_indices shape: (batch_size, num_zones) - now contains all zone indices
        # static_features_all_zones shape: (num_zones, num_static_features)
        # haversine_distance_matrix shape: (num_zones, num_zones)

        b, s, n, d_ts = dynamic_features.shape # b=batch_size, s=seq_len, n=num_zones, d_ts=num_ts_features
        logger.debug(f"DHKINet.forward - Input dynamic_features shape: {dynamic_features.shape}")
        logger.debug(f"DHKINet.forward - Input zone_indices shape: {zone_indices.shape}")
        logger.debug(f"DHKINet.forward - Input static_features_all_zones shape: {static_features_all_zones.shape}")
        logger.debug(f"DHKINet.forward - Input haversine_distance_matrix shape: {haversine_distance_matrix.shape}")

        # --- Observational Stream ---
        # 1. Get learnable zone embeddings for all zones
        # (num_zones, d_learnable) -> (1, 1, num_zones, d_learnable) -> (b, s, num_zones, d_learnable)
        learnable_zone_embeds = self.phi_emb(zone_indices[0]).unsqueeze(0).unsqueeze(0).expand(b, s, -1, -1)
        logger.debug(f"DHKINet.forward - learnable_zone_embeds shape: {learnable_zone_embeds.shape}")

        # 2. Concatenate dynamic features with learnable zone embeddings
        # (b, s, n, d_ts + d_learnable)
        x_combined = torch.cat([dynamic_features, learnable_zone_embeds], dim=-1)
        logger.debug(f"DHKINet.forward - x_combined shape: {x_combined.shape}")

        # 3. Project the combined features into the model's dimension
        # Reshape for projection: (batch * seq_len * num_zones, d_ts + d_learnable)
        x_projected = self.input_projection(x_combined.view(b * s * n, -1))
        # Reshape back: (b, s, n, model_dim)
        x_projected = x_projected.view(b, s, n, -1)
        logger.debug(f"DHKINet.forward - x_projected shape: {x_projected.shape}")

        # 4. Prepare for Transformer: (batch_size * seq_len, num_zones, model_dim)
        # Reshape to (b * s, n, model_dim)
        h_obs_input = x_projected.view(b * s, n, -1)
        logger.debug(f"DHKINet.forward - h_obs_input (before Transformer) shape: {h_obs_input.shape}")

        # 5. Get relative position bias matrix
        if self.disable_spatial_bias:
            pos_indices = torch.arange(n, device=dynamic_features.device).unsqueeze(0).expand(b * s, -1)
            h_obs_input = h_obs_input + self.positional_encoding(pos_indices)
            bias_matrix = None
        else:
            if haversine_distance_matrix.numel() > 0:
                bias_matrix = self.relative_position_bias(haversine_distance_matrix)
                bias_matrix = bias_matrix.repeat(s, 1, 1)
                logger.debug(f"DHKINet.forward - bias_matrix (repeated for seq_len) shape: {bias_matrix.shape}")
            else:
                bias_matrix = None
                logger.debug("DHKINet.forward - haversine_distance_matrix is empty, bias_matrix set to None.")

        # 6. Feed into Transformer Encoder layers
        for i, layer in enumerate(self.transformer_encoder_layers):
            h_obs_input = layer(h_obs_input, bias_matrix=bias_matrix)
            logger.debug(f"DHKINet.forward - h_obs_input (after Transformer layer {i}) shape: {h_obs_input.shape}")
        h_obs_output = self.transformer_norm(h_obs_input)
        logger.debug(f"DHKINet.forward - h_obs_output (after Transformer norm) shape: {h_obs_output.shape}")
        
        # Reshape h_obs_output back to (b, s, n, model_dim)
        h_obs_output_reshaped = h_obs_output.view(b, s, n, -1)
        logger.debug(f"DHKINet.forward - h_obs_output_reshaped shape: {h_obs_output_reshaped.shape}")

        # Take the output of the last time step for prediction
        # Shape: (batch_size, num_zones, model_dim)
        h_obs = h_obs_output_reshaped[:, -1, :, :]
        logger.debug(f"DHKINet.forward - h_obs (last time step) shape: {h_obs.shape}")

        # --- Contextual Stream ---
        # static_features_all_zones already has the batch dimension from the dataloader.
        # Shape: (batch_size, num_zones, num_static_features)
        h_ctx_all_zones = self.contextual_stream(static_features_all_zones)
        logger.debug(f"DHKINet.forward - h_ctx_all_zones shape: {h_ctx_all_zones.shape}")

        # --- Hyper-Knowledge Infusion (Gated Fusion) ---
        h_fused = self.fusion_module(h_obs, h_ctx_all_zones)
        logger.debug(f"DHKINet.forward - h_fused shape: {h_fused.shape}")

        # --- Prediction Heads ---
        if direct_income_prediction:
            pred_total_income = self.direct_income_head(h_fused)
            outputs = {"pred_total_income": pred_total_income}
        else:
            # Create specialized input for the Value Head
            h_value = torch.cat([h_fused, static_features_all_zones], dim=-1)
            logger.debug(f"DHKINet.forward - h_value (for Value Head) shape: {h_value.shape}")

            has_orders_logits = self.classification_head(h_fused)
            log_trip_count_pred = self.trip_count_head(h_fused)
            log_avg_income_pred = self.avg_income_head(h_value)
            logger.debug(f"DHKINet.forward - has_orders_logits shape: {has_orders_logits.shape}")
            logger.debug(f"DHKINet.forward - log_trip_count_pred shape: {log_trip_count_pred.shape}")
            logger.debug(f"DHKINet.forward - log_avg_income_pred shape: {log_avg_income_pred.shape}")
            outputs = {
                "has_orders_logits": has_orders_logits,
                "log_trip_count_pred": log_trip_count_pred,
                "log_avg_income_pred": log_avg_income_pred,
            }

        # --- Contrastive Learning Outputs (always computed for potential loss) ---
        # Pass through contextual stream to get embeddings for contrastive loss
        anchor_embedding = self.contextual_stream(static_features_anchor.unsqueeze(0))
        positive_embedding = self.contextual_stream(static_features_positive.unsqueeze(0))
        negative_embeddings = self.contextual_stream(static_features_negatives.unsqueeze(0))
        logger.debug(f"DHKINet.forward - anchor_embedding shape: {anchor_embedding.shape}")
        logger.debug(f"DHKINet.forward - positive_embedding shape: {positive_embedding.shape}")
        logger.debug(f"DHKINet.forward - negative_embeddings shape: {negative_embeddings.shape}")

        # Add contrastive embeddings to the output dictionary
        outputs.update({
            "anchor_embedding": anchor_embedding.squeeze(0),
            "positive_embedding": positive_embedding.squeeze(0),
            "negative_embeddings": negative_embeddings.squeeze(0)
        })

        return outputs