import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,SGConv,GraphConv


class GATv2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers, heads=8):
        torch.manual_seed(12345)
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(dim_in, dim_h, heads=heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(dim_h * heads, dim_h, heads=heads))
            
        # Last layer
        self.convs.append(GATv2Conv(dim_h * heads, dim_out, heads=heads))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class SimpleGCNRes(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Input projection if dimensions don't match
        self.input_proj = None
        if num_features != hidden_channels:
            self.input_proj = nn.Linear(num_features, hidden_channels)
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        # Handle first layer separately for residual connection
        identity = x
        x = self.convs[0](x, edge_index)
        x = x.relu()
        
        # Project input if dimensions don't match
        if self.input_proj is not None:
            identity = self.input_proj(identity)
        
        x = x + identity  # First residual connection
        x = F.dropout(x, p=0.0, training=self.training)
        
        # Middle layers with residual connections
        for conv in self.convs[1:-1]:
            identity = x
            x = conv(x, edge_index)
            x = x.relu()
            x = x + identity  # Residual connection
            x = F.dropout(x, p=0.0, training=self.training)
        
        # Final layer without residual connection
        x = self.convs[-1](x, edge_index)
        return x

