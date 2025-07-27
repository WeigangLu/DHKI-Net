import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    """
    Implements a simple and efficient Gated Fusion mechanism.
    The formula is: H_final = g * H_obs + (1 - g) * H_ctx
    where g is the sigmoid gate, H_obs is the observational stream output,
    and H_ctx is the contextual stream output.
    """
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        # The gate takes concatenated observational and contextual features
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )

    def forward(self, h_obs: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        """
        Fuses the observational and contextual stream outputs using a learned gate.

        Args:
            h_obs (torch.Tensor): Output from the observational stream.
                                  Shape: (batch_size, num_zones, model_dim)
            h_ctx (torch.Tensor): Output from the contextual stream.
                                  Shape: (batch_size, num_zones, model_dim)

        Returns:
            torch.Tensor: The fused representation. Shape: (batch_size, num_zones, model_dim)
        """        
        # Concatenate the two streams to form the input for the gate
        gate_input = torch.cat([h_obs, h_ctx], dim=-1)
        
        # Calculate the gate values
        g = self.gate(gate_input)
        
        # Apply the gated fusion formula
        fused_representation = g * h_obs + (1 - g) * h_ctx
        
        return fused_representation
