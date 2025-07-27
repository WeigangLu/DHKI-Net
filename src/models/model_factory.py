
import torch.nn as nn
from typing import Dict

from models.components import RelativePositionBias, TransformerEncoderLayer, ClassificationHead, RegressionHead, AttentionHead
from models.contextual_stream import ContextualStream
from models.fusion_module import GatedFusion
from models.fusion_module_alternatives import AdditiveAttentionFusion, ParameterizedBilinearFusion
from models.dhk_net import DHKINet

def create_model(config: Dict, num_zones: int, num_ts_features: int, disable_spatial_bias: bool = False, value_head_type: str = 'attention') -> DHKINet:
    """Factory function to create the DHKI-Net model with specified components."""
    model_config = config['model']
    
    # --- Observational Stream ---
    obs_stream_type = model_config.get('observational_stream_type', 'transformer')
    if obs_stream_type == 'transformer':
        observational_stream = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=model_config['embedding_dim'],
                num_heads=model_config['num_heads'],
                ff_dim=model_config['embedding_dim'] * 4,
                dropout_rate=model_config['dropout_rate']
            )
            for _ in range(model_config['num_transformer_layers'])
        ])
    else:
        raise ValueError(f"Unknown observational_stream_type: {obs_stream_type}")

    # --- Contextual Stream ---
    contextual_stream_type = model_config.get('contextual_stream_type', 'mlp')
    if contextual_stream_type == 'mlp':
        contextual_stream = ContextualStream(config)
    else:
        raise ValueError(f"Unknown contextual_stream_type: {contextual_stream_type}")

    # --- Fusion Module ---
    fusion_module_type = model_config.get('fusion_module_type', 'gated')
    if fusion_module_type == 'gated':
        fusion_module = GatedFusion(model_config['embedding_dim'])
    elif fusion_module_type == 'additive':
        fusion_module = AdditiveAttentionFusion(model_config['embedding_dim'])
    elif fusion_module_type == 'bilinear':
        fusion_module = ParameterizedBilinearFusion(model_config['embedding_dim'])
    else:
        raise ValueError(f"Unknown fusion_module_type: {fusion_module_type}")

    # --- Prediction Heads ---
    num_static_features = config['data']['num_static_features']
    classification_head = ClassificationHead(model_config['embedding_dim'])
    trip_count_head = RegressionHead(model_config['embedding_dim'])
    if value_head_type == 'attention':
        avg_income_head = AttentionHead(input_dim=model_config['embedding_dim'] + num_static_features)
    elif value_head_type == 'mlp':
        avg_income_head = RegressionHead(embed_dim=model_config['embedding_dim'] + num_static_features)
    else:
        raise ValueError(f"Unknown value_head_type: {value_head_type}")

    # --- DHKI-Net ---
    model = DHKINet(
        config=config,
        num_zones=num_zones,
        num_ts_features=num_ts_features,
        observational_stream=observational_stream,
        contextual_stream=contextual_stream,
        fusion_module=fusion_module,
        classification_head=classification_head,
        trip_count_head=trip_count_head,
        avg_income_head=avg_income_head,
        disable_spatial_bias=disable_spatial_bias
    )

    return model
