import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class STGCN_Baseline(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network (STGCN) baseline model.
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, direct_income=False):
        super(STGCN_Baseline, self).__init__()
        self.temporal_conv1 = nn.Conv2d(num_features, 32, kernel_size=(3, 1), padding=(1, 0))
        self.graph_conv1 = GCNConv(32, 64)
        self.temporal_conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(128, 1)
        else:
            self.fc_count = nn.Linear(128, 1)
            self.fc_income = nn.Linear(128, 1)

    def forward(self, x, edge_index):
        b, t, n, f = x.shape
        x = x.permute(0, 3, 1, 2) # (b, f, t, n)
        x = self.temporal_conv1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1).reshape(b * t, n, -1)
        x = self.graph_conv1(x, edge_index)
        x = torch.relu(x)
        x = x.reshape(b, t, n, -1).permute(0, 3, 1, 2)
        x = self.temporal_conv2(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1).reshape(b, t, n, -1)
        x = x[:, -1, :, :] # Take last time step, shape (b, n, 128)
        if self.direct_income:
            return self.fc(x).squeeze(-1)
        else:
            count_pred = self.fc_count(x).squeeze(-1)
            income_pred = self.fc_income(x).squeeze(-1)
            return count_pred, income_pred

class DCGRUCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_proj):
        super(DCGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim * 2)
        self.update_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.diffusion_conv = nn.Linear(num_nodes, num_nodes * num_proj)

    def forward(self, x, h, adj):
        combined = torch.cat([x, h], dim=2)
        gates = torch.sigmoid(self.gate_proj(combined))
        r, u = gates.chunk(2, dim=2)
        c = torch.tanh(self.update_proj(torch.cat([x, r * h], dim=2)))
        c_diff = self.diffusion_conv(c.permute(0, 2, 1)).permute(0, 2, 1)
        h_new = u * h + (1 - u) * c_diff
        return h_new

class DCRNN_Baseline(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, num_layers, num_timesteps_output, direct_income=False):
        super(DCRNN_Baseline, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(num_nodes, num_features, hidden_dim, 1))
        for _ in range(1, num_layers):
            self.cells.append(DCGRUCell(num_nodes, hidden_dim, hidden_dim, 1))
        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        b, t, n, f = x.shape
        h = torch.zeros(b, n, self.hidden_dim, device=x.device)
        for i in range(t):
            h_layer = x[:, i, :, :]
            for j in range(self.num_layers):
                h = self.cells[j](h_layer, h, adj)
                h_layer = h
        if self.direct_income:
            return self.fc(h).squeeze(-1)
        else:
            count_pred = self.fc_count(h).squeeze(-1)
            income_pred = self.fc_income(h).squeeze(-1)
            return count_pred, income_pred

class LSTM_GAT_Baseline(nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int = 64, num_lstm_layers: int = 1, num_gat_layers: int = 1, direct_income=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=num_input_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        self.gat_layers = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim))
        self.direct_income = direct_income
        if direct_income:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc_count = nn.Linear(hidden_dim, 1)
            self.fc_income = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic_features: torch.Tensor, edge_index: torch.Tensor):
        batch_size, seq_len, num_zones, num_features = dynamic_features.shape
        lstm_input = dynamic_features.permute(0, 2, 1, 3).reshape(batch_size * num_zones, seq_len, num_features)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_output_reshaped = lstm_out[:, -1, :]
        gat_input = lstm_output_reshaped.reshape(batch_size, num_zones, self.hidden_dim)
        gat_output_list = []
        for i in range(batch_size):
            x = gat_input[i]
            for gat_layer in self.gat_layers:
                x = gat_layer(x, edge_index)
                x = torch.relu(x)
            gat_output_list.append(x)
        gat_output = torch.stack(gat_output_list, dim=0)
        if self.direct_income:
            return self.fc(gat_output).squeeze(-1)
        else:
            count_pred = self.fc_count(gat_output).squeeze(-1)
            income_pred = self.fc_income(gat_output).squeeze(-1)
            return count_pred, income_pred
