import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import GATConv


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.sample_up_1 = Linear(1, 16)
        self.sample_up_2 = Linear(16, 32)
        self.sample_up_3 = Linear(32, 64)
        self.conv1 = GATConv(64, 128)
        self.conv2 = GATConv(128, 256)
        self.conv3 = GATConv(256, 128)
        self.conv4 = GATConv(128, 64)
        self.conv5 = GATConv(64, 32)
        self.lin = Linear(32, 1)

    def forward(self, x, edge_index, batch):
        x = self.sample_up_1(x).relu()
        x = self.sample_up_2(x).relu()
        x = self.sample_up_3(x).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x).sigmoid()
        return x

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = Linear(28, 64)
        self.lin2 = Linear(64, 16)
        self.lin3 = Linear(16, 1)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).sigmoid()
        return x