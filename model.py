import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, ARMAConv, GATv2Conv, SortAggregation
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import aggr
from torch_geometric.transforms import GDC


class GNN(torch.nn.Module):
# TODO rethink the model, needs some better message passing
    def __init__(self):
        super(GNN, self).__init__()
        self.gat_conv1 = GATConv(1, 64)
        self.graph_norm_1 = GraphNorm(64)
        self.gat_conv2 = GATConv(64, 512)
        self.graph_norm_2 = GraphNorm(512)
        self.gat_conv3 = GATConv(512, 2048)
        self.graph_norm_3 = GraphNorm(2048)
        self.gcn_conv1 = GCNConv(2048, 512)
        self.graph_norm_4 = GraphNorm(512)
        self.gcn_conv2 = GCNConv(512, 64)
        self.graph_norm_5 = GraphNorm(64)
        self.lin1 = Linear(64*3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_1(x, batch)
        x = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_2(x, batch)
        x = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_3(x, batch)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_4(x, batch)
        x = self.gcn_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_5(x, batch)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_add], dim=1)
        # x = self.aggr(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

class GNN_v2(torch.nn.Module):
    def __init__(self):
        super(GNN_v2, self).__init__()
        self.gdc1 = GDC()
        self.gat_conv1 = GATConv(1, 16)
        self.gcn_conv1 = GCNConv(16, 32)
        self.lin1 = Linear(32*3, 2)
    def forward(self, data):
        new_data = self.gdc1(data)
        x, edge_index, batch = new_data.x, new_data.edge_index, new_data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        # x = self.gat_conv2(x, edge_index)
        # x = F.elu(x)
        # x = self.gat_conv3(x, edge_index)
        # x = F.elu(x)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_add], dim=1)
        x = self.lin1(x)
        return x

class GNN_v3(torch.nn.Module):
    def __init__(self):
        super(GNN_v3, self).__init__()
        self.gat_conv1 = GATConv(1, 64)
        self.graph_norm_1 = GraphNorm(64)
        self.gat_conv2 = GATConv(64, 512)
        self.graph_norm_2 = GraphNorm(512)
        self.gat_conv3 = GATConv(512, 2048)
        self.graph_norm_3 = GraphNorm(2048)
        self.gcn_conv1 = ARMAConv(2048, 512)
        self.graph_norm_4 = GraphNorm(512)
        self.gcn_conv2 = ARMAConv(512, 64)
        self.graph_norm_5 = GraphNorm(64)
        # self.aggr = aggr.MLPAggregation(64, 64, 7, num_layers=1)
        self.aggr = aggr.SortAggregation(5)
        self.lin1 = Linear(64*8, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_1(x, batch)
        x = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_2(x, batch)
        x = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_3(x, batch)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_4(x, batch)
        x = self.gcn_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_5(x, batch)
        x_aggr = self.aggr(x, batch)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_add, x_aggr], dim=1)
        # x = self.aggr(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

class GNN_v3_mini(torch.nn.Module):
    def __init__(self):
        super(GNN_v3_mini, self).__init__()
        self.gat_conv1 = GATConv(1, 32)
        self.graph_norm_1 = GraphNorm(32)
        self.gat_conv2 = GATConv(32, 64)
        self.graph_norm_2 = GraphNorm(64)
        self.gat_conv3 = GATConv(64, 256)
        self.graph_norm_3 = GraphNorm(256)
        self.gcn_conv1 = ARMAConv(256, 64)
        self.graph_norm_4 = GraphNorm(64)
        self.gcn_conv2 = ARMAConv(64, 32)
        self.graph_norm_5 = GraphNorm(32)
        # self.aggr = aggr.MLPAggregation(64, 64, 7, num_layers=1)
        # self.aggr = aggr.SortAggregation(5)
        self.lin1 = Linear(32*3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_1(x, batch)
        x = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_2(x, batch)
        x = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_3(x, batch)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_4(x, batch)
        x = self.gcn_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_5(x, batch)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_add], dim=1)
        # x = self.aggr(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

class GNN_v4(torch.nn.Module):
    def __init__(self):
        super(GNN_v4, self).__init__()
        self.gat_conv1 = GATv2Conv(1, 64)
        self.graph_norm_1 = GraphNorm(64)
        self.gat_conv2 = GATv2Conv(64, 512)
        self.graph_norm_2 = GraphNorm(512)
        self.gat_conv3 = GATv2Conv(512, 2048)
        self.graph_norm_3 = GraphNorm(2048)
        self.gcn_conv1 = ARMAConv(2048, 512)
        self.graph_norm_4 = GraphNorm(512)
        self.gcn_conv2 = ARMAConv(512, 64)
        self.graph_norm_5 = GraphNorm(64)
        self.lin1 = Linear(64*3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_1(x, batch)
        x = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_2(x, batch)
        x = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_3(x, batch)
        x = self.gcn_conv1(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_4(x, batch)
        x = self.gcn_conv2(x, edge_index)
        x = F.elu(x)
        x = self.graph_norm_5(x, batch)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_add], dim=1)
        # x = self.aggr(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

class GNN_v5(torch.nn.Module):
    # An End-to-End Deep Learning Architecture for Graph Classification
    def __init__(self):
        super(GNN_v5, self).__init__()
        self.gat_conv1 = GATv2Conv(1, 64)
        self.gat_conv2 = GATv2Conv(64, 128)
        self.gcn_conv1 = ARMAConv(1, 64)
        self.gcn_conv2 = ARMAConv(64, 128)
        self.aggr = SortAggregation(10)
        self.lin1 = Linear(128*3*2, 64)
        self.lin2 = Linear(128*10*2, 128)
        self.lin3 = Linear(192, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        gat_conv1 = self.gat_conv1(x, edge_index)
        gat_conv1 = F.elu(gat_conv1)

        gat_conv2 = self.gat_conv2(gat_conv1, edge_index)
        gat_conv2 = F.elu(gat_conv2)

        gcn_conv1 = self.gcn_conv1(x, edge_index)
        gcn_conv1 = F.elu(gcn_conv1)

        gcn_conv2 = self.gcn_conv2(gcn_conv1, edge_index)
        gcn_conv2 = F.elu(gcn_conv2)

        gcn_gat = torch.cat([gat_conv2, gcn_conv2], dim=1)

        x_max = global_max_pool(gcn_gat, batch)
        x_mean = global_mean_pool(gcn_gat, batch)
        x_add = global_add_pool(gcn_gat, batch)
        x_norm_pool = torch.cat([x_max, x_mean, x_add], dim=1)
        x_norm_pool = self.lin1(x_norm_pool)

        x_agg_pool = self.aggr(gcn_gat, batch)
        # x = torch.cat([x, x_agg], dim=1)
        x_agg_pool = self.lin2(x_agg_pool)

        x = torch.cat([x_norm_pool, x_agg_pool], dim=1)
        x = self.lin3(x)
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