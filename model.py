import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn import aggr


class GNN(torch.nn.Module):
# TODO rethink the model, needs some better message passing
    def __init__(self):
        super(GNN, self).__init__()
        self.gat_conv1 = GATConv(1, 128)
        self.gat_conv2 = GATConv(128, 128)
        self.gat_conv3 = GATConv(128, 128)
        self.gcn_conv1 = GCNConv(128, 64)
        self.gcn_conv2 = GCNConv(64, 32)
        self.aggr = aggr.SoftmaxAggregation(learn=True)
        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.elu(x)
        # x = global_mean_pool(x, batch)
        x = self.aggr(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)
        return x

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = Linear(28, 64)
        self.lin2 = Linear(64, 256)
        self.lin3 = Linear(256, 64)
        self.lin4 = Linear(64, 28)
        self.lin5 = Linear(28, 1)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).relu()
        x = self.lin4(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x).sigmoid()
        return x