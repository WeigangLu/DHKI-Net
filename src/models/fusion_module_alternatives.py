import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim * 2, 1)

    def forward(self, h_obs: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        # Concatenate the two representations
        combined = torch.cat([h_obs, h_ctx], dim=-1)
        # Calculate attention scores
        attn_scores = self.attention_weights(combined)
        attn_weights = F.softmax(attn_scores, dim=1)
        # Apply attention weights
        fused = attn_weights * h_obs + (1 - attn_weights) * h_ctx
        return fused

class ParameterizedBilinearFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, embed_dim)

    def forward(self, h_obs: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        return self.bilinear(h_obs, h_ctx)