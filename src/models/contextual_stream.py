#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implements the Contextual Stream of the DHKI-Net architecture.
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ContextualStream(nn.Module):
    """
    Implements the Contextual Stream for learning stable region embeddings.

    This stream takes static regional features (e.g., POI distributions, road network
    statistics) and learns a deep, non-linear representation for each region.
    The primary training mechanism for this stream is the InfoNCE loss (a form of contrastive loss),
    which helps in learning an embedding space where similar regions are closer together.
    It uses in-batch negative sampling. More advanced sampling like hard negative mining
    is a potential future improvement.

    The encoder can be a simple MLP or a more complex architecture depending on the
    nature of the static features.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initializes the ContextualStream module.

        Args:
            config (Dict): A configuration dictionary, containing parameters
                         for the encoder, such as hidden layer sizes or dropout rates.
        """
        super().__init__()
        self.feature_dim = config['data']['num_static_features'] # Assuming num_static_features is available in config
        self.embedding_dim = config['model']['embedding_dim']
        logger.info(f"ContextualStream initialized with feature_dim: {self.feature_dim}, embedding_dim: {self.embedding_dim}")

        # --- Define the Encoder ---
        # A simple MLP is used here as a starting point. This can be replaced
        # with more sophisticated architectures like ResNet blocks or Attention networks.
        hidden_dim = config.get('contextual_stream_hidden_dim', 512)
        dropout_rate = config.get('contextual_stream_dropout', 0.1)

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )

        # The final output is NOT normalized here. Normalization (e.g., L2 norm)
        # is typically applied right before the contrastive loss calculation.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Static features tensor.

        Returns:
            torch.Tensor: Contextual embedding.
        """
        logger.debug(f"ContextualStream input shape: {x.shape}")
        embeddings = self.encoder(x)
        return embeddings