import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GATConv


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GATConv(1, 64)
        self.conv2 = GATConv(64, 32)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
         # Choose between different GNN building blocks:
        x = global_mean_pool(x, batch)
        return self.lin(x)