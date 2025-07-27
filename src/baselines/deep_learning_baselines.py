import torch
import torch.nn as nn
import logging
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)

class BaseDeepLearningBaseline(nn.Module):
    """
    Base class for deep learning baselines.
    Provides common functionalities like handling input features and prediction heads.
    """
    def __init__(self, num_input_features: int, hidden_dim: int, num_output_features: int = 2):
        super().__init__()
        self.num_input_features = num_input_features
        self.hidden_dim = hidden_dim
        self.num_output_features = num_output_features # trip_count, average_income

        # Define prediction heads for trip_count and average_income
        self.count_head = nn.Linear(self.hidden_dim, 1)
        self.income_head = nn.Linear(self.hidden_dim, 1)

class LSTM_Baseline(nn.Module):
    def __init__(self, num_features, hidden_dim, direct_income=False):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, num_zones, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_zones, seq_len, num_features)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        if self.direct_income:
            return self.fc(last_hidden_state).reshape(batch_size, num_zones)
        else:
            pred_count = self.fc_count(last_hidden_state)
            pred_income = self.fc_income(last_hidden_state)
            return pred_count.reshape(batch_size, num_zones), pred_income.reshape(batch_size, num_zones)

class RNN_Baseline(nn.Module):
    def __init__(self, num_features, hidden_dim, direct_income=False):
        super().__init__()
        self.rnn = nn.RNN(num_features, hidden_dim, batch_first=True)
        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, num_zones, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_zones, seq_len, num_features)
        rnn_out, _ = self.rnn(x)
        last_hidden_state = rnn_out[:, -1, :]
        if self.direct_income:
            return self.fc(last_hidden_state).reshape(batch_size, num_zones)
        else:
            pred_count = self.fc_count(last_hidden_state)
            pred_income = self.fc_income(last_hidden_state)
            return pred_count.reshape(batch_size, num_zones), pred_income.reshape(batch_size, num_zones)

class LSTM_GCN_Baseline(BaseDeepLearningBaseline):
    """
    Combines LSTM for temporal feature extraction with GCN for spatial aggregation.
    """
    def __init__(self, num_input_features: int, hidden_dim: int = 64, num_lstm_layers: int = 1, num_gcn_layers: int = 1, num_output_features: int = 2, direct_income=False):
        super().__init__(num_input_features, hidden_dim, num_output_features)
        
        self.lstm = nn.LSTM(
            input_size=num_input_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic_features: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # dynamic_features: (B, T_hist, N, F_dynamic)
        batch_size, seq_len, num_zones, num_features = dynamic_features.shape
        
        # Process each zone's time series with LSTM
        # Reshape to (B * N, T_hist, F_dynamic)
        lstm_input = dynamic_features.permute(0, 2, 1, 3).reshape(batch_size * num_zones, seq_len, num_features)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Take the output from the last time step
        lstm_output_reshaped = lstm_out[:, -1, :].reshape(batch_size, num_zones, self.hidden_dim) # (B, N, hidden_dim)
        
        # Apply GCN layer(s)
        # GCN expects (num_nodes, num_features) and edge_index
        # We need to process each batch independently for GCN
        gcn_output_list = []
        for i in range(batch_size):
            x = lstm_output_reshaped[i] # (N, hidden_dim)
            # Ensure edge_index is on the same device as x
            current_edge_index = edge_index.to(x.device)
            for gcn_layer in self.gcn_layers:
                x = gcn_layer(x, current_edge_index)
                x = torch.relu(x) # Apply activation after GCN
            gcn_output_list.append(x)
        
        gcn_output = torch.cat(gcn_output_list, dim=0) # (B * N, hidden_dim)
        
        if self.direct_income:
            return self.fc(gcn_output).reshape(batch_size, num_zones)
        else:
            # Predict for count and income
            count_pred = self.fc_count(gcn_output) # (B * N, 1)
            income_pred = self.fc_income(gcn_output) # (B * N, 1)
            
            # Reshape back to (B, N) for consistency with targets
            count_pred = count_pred.reshape(batch_size, num_zones)
            income_pred = income_pred.reshape(batch_size, num_zones)
            
            return count_pred, income_pred

class Transformer_Baseline(BaseDeepLearningBaseline):
    """
    Transformer-based baseline model for spatio-temporal prediction.
    Processes dynamic features for all zones at each time step.
    """
    def __init__(self, num_input_features: int, hidden_dim: int = 64, num_heads: int = 1, num_layers: int = 1, num_output_features: int = 2, max_seq_len: int = 48, num_zones: int = 450, direct_income=False):
        super().__init__(num_input_features, hidden_dim, num_output_features)
        
        self.embedding_layer = nn.Linear(num_input_features * num_zones, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim)) # (1, T_hist, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic_features: torch.Tensor, edge_index: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # dynamic_features: (B, T_hist, N, F_dynamic)
        batch_size, seq_len, num_zones, num_features = dynamic_features.shape
        
        # Reshape to (B, T_hist, N * F_dynamic) for Transformer processing
        # Or, process each zone independently and then combine? Let's try combining first.
        # For a simple baseline, we can flatten N and F_dynamic into one feature dimension
        # (B, T_hist, N * F_dynamic)
        transformer_input = dynamic_features.reshape(batch_size, seq_len, num_zones * num_features)
        
        # Embed the input features
        embedded_input = self.embedding_layer(transformer_input) # (B, T_hist, hidden_dim)
        
        # Add positional encoding
        # Ensure positional_encoding matches the sequence length
        if seq_len > self.positional_encoding.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.positional_encoding.shape[1]}.")
        
        transformer_input_with_pos = embedded_input + self.positional_encoding[:, :seq_len, :]
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input_with_pos) # (B, T_hist, hidden_dim)
        
        # Take the output from the last time step
        last_time_step_output = transformer_output[:, -1, :] # (B, hidden_dim)
        
        if self.direct_income:
            return self.fc(last_time_step_output).repeat(1, num_zones)
        else:
            # Predict for count and income
            # This prediction is per-batch, not per-zone. We need to adapt.
            # For a simple baseline, we can predict a single value for the entire batch
            # and then expand it to all zones. This is a simplification.
            # A more complex Transformer would need to handle zone-level predictions.
            count_pred_batch = self.fc_count(last_time_step_output) # (B, 1)
            income_pred_batch = self.fc_income(last_time_step_output) # (B, 1)
            
            # Expand predictions to (B, N) for consistency with targets
            count_pred = count_pred_batch.repeat(1, num_zones)
            income_pred = income_pred_batch.repeat(1, num_zones)
            
            return count_pred, income_pred