import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)

class RelativePositionBias(nn.Module):
    """
    Injects a distance-aware bias directly into the Transformer's attention mechanism.
    Maps a continuous distance into one of K discrete distance buckets and learns a scalar bias for each bucket.
    """
    def __init__(self, num_buckets: int):
        super().__init__()
        # phi_dist: maps each distance bucket to a learnable scalar bias value.
        # The embedding dimension is 1 because we need a scalar bias.
        self.phi_dist = nn.Embedding(num_buckets, 1)

    def forward(self, bucketized_distance_matrix):
        # Ensure bucket indices are within valid range
        bucketized_distance_matrix = bucketized_distance_matrix.clamp(0, self.phi_dist.num_embeddings - 1)
        # Convert to long integer type and get embeddings
        bias_matrix = self.phi_dist(bucketized_distance_matrix.long()).squeeze(-1)
        return bias_matrix

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional relative position bias.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor,
                bias_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            bias_matrix (torch.Tensor, optional): Spatial bias matrix of shape (batch_size, seq_len, seq_len)
                                                 to be added to attention logits.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()

        # Project and reshape for multi-head attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logger.debug(f"MultiHeadSelfAttention - q shape: {q.shape}")
        logger.debug(f"MultiHeadSelfAttention - k shape: {k.shape}")
        logger.debug(f"MultiHeadSelfAttention - v shape: {v.shape}")

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logger.debug(f"MultiHeadSelfAttention - attention_scores shape: {attention_scores.shape}")

        if bias_matrix is not None:
            logger.debug(f"MultiHeadSelfAttention - bias_matrix shape: {bias_matrix.shape}")
            # The bias_matrix is (batch_size * seq_len, num_zones, num_zones)
            # We need to add a head dimension to it.
            # Reshape to (batch_size, seq_len, num_zones, num_zones)
            bias_matrix = bias_matrix.view(batch_size, seq_len, seq_len, -1).squeeze(-1)
            # Add head dimension: (batch_size, 1, seq_len, seq_len)
            bias_matrix = bias_matrix.unsqueeze(1)
            logger.debug(f"MultiHeadSelfAttention - reshaped bias_matrix shape: {bias_matrix.shape}")
            attention_scores = attention_scores + bias_matrix

        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_output = torch.matmul(attention_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer with Multi-Head Self-Attention and Feed-Forward Network.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, bias_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attn(x, bias_matrix=bias_matrix)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.mlp(x).squeeze(-1)

class AttentionHead(nn.Module):
    """A more powerful prediction head using self-attention."""
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        # x shape: (batch_size, num_zones, embed_dim)
        # Self-attention expects (seq_len, batch_size, embed_dim) if batch_first=False
        # or (batch_size, seq_len, embed_dim) if batch_first=True.
        # Here, num_zones is our "sequence".
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        return self.ffn(x).squeeze(-1)