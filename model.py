import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear, ReLU, ELU
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, ARMAConv, GATv2Conv, SortAggregation, \
    MLPAggregation, PointNetConv, TopKPooling, GCN, GAT
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import aggr
from torch_geometric.transforms import GDC
from torch_geometric.utils import erdos_renyi_graph


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
        self.lin1 = Linear(64 * 3, 2)

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
        self.lin1 = Linear(32 * 3, 2)

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
        self.lin1 = Linear(64 * 8, 2)

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
        self.lin1 = Linear(32 * 3, 2)

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
        self.lin1 = Linear(64 * 3, 2)

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
    # Jet index encoding?
    # An End-to-End Deep Learning Architecture for Graph Classification
    def __init__(self):
        super(GNN_v5, self).__init__()
        out_hidden = 512
        self.gat_conv1 = GATv2Conv(1, 64)
        self.gat_conv1_norm = GraphNorm(64)
        self.gat_conv2 = GATv2Conv(64, 128)
        self.gat_conv2_norm = GraphNorm(128)
        self.gat_conv3 = GATv2Conv(128, out_hidden)
        self.gat_conv3_norm = GraphNorm(out_hidden)
        self.gcn_conv1 = ARMAConv(1, 64)
        self.gcn_conv1_norm = GraphNorm(64)
        self.gcn_conv2 = ARMAConv(64, 128)
        self.gcn_conv2_norm = GraphNorm(128)
        self.gcn_conv3 = ARMAConv(128, out_hidden)
        self.gcn_conv3_norm = GraphNorm(out_hidden)
        self.aggr = SortAggregation(12)
        self.lin1 = Linear(out_hidden * 3 * 2, out_hidden)
        self.lin2 = Linear(out_hidden * 12 * 2, out_hidden)

        self.lin3 = Linear(out_hidden * 2, 1, bias=False)
        # self.lin4 = Linear(2, 2, bias=False) 

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        gat_conv1 = self.gat_conv1(x, edge_index)
        gat_conv1 = F.elu(gat_conv1)
        gat_conv1 = self.gat_conv1_norm(gat_conv1, batch)

        gat_conv2 = self.gat_conv2(gat_conv1, edge_index)
        gat_conv2 = F.elu(gat_conv2)
        gat_conv2 = self.gat_conv2_norm(gat_conv2, batch)

        gat_conv3 = self.gat_conv3(gat_conv2, edge_index)
        gat_conv3 = F.elu(gat_conv3)
        gat_conv3 = self.gat_conv3_norm(gat_conv3, batch)

        gcn_conv1 = self.gcn_conv1(x, edge_index)
        gcn_conv1 = F.elu(gcn_conv1)
        gcn_conv1 = self.gcn_conv1_norm(gcn_conv1, batch)

        gcn_conv2 = self.gcn_conv2(gcn_conv1, edge_index)
        gcn_conv2 = F.elu(gcn_conv2)
        gcn_conv2 = self.gcn_conv2_norm(gcn_conv2, batch)

        gcn_conv3 = self.gcn_conv3(gcn_conv2, edge_index)
        gcn_conv3 = F.elu(gcn_conv3)
        gcn_conv3 = self.gcn_conv3_norm(gcn_conv3, batch)

        gcn_gat = torch.cat([gat_conv3, gcn_conv3], dim=1)

        x_max = global_max_pool(gcn_gat, batch)
        x_mean = global_mean_pool(gcn_gat, batch)
        x_add = global_add_pool(gcn_gat, batch)
        x_norm_pool = torch.cat([x_max, x_mean, x_add], dim=1)
        x_norm_pool = self.lin1(x_norm_pool)

        x_agg_pool = self.aggr(gcn_gat, batch)
        x_agg_pool = self.lin2(x_agg_pool)

        x = torch.cat([x_norm_pool, x_agg_pool], dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        # x = self.lin4(x)
        return x


class GNN_v6(torch.nn.Module):
    def __init__(self):
        super(GNN_v6, self).__init__()

        self.local_nn_1 = torch.nn.Sequential(
            Linear(3 * 2, 32),
            ELU(),
            Linear(32, 32),
            ELU(),
            Linear(32, 32),
        )

        self.global_nn_1 = torch.nn.Sequential(
            Linear(32, 32),
            ELU(),
            Linear(32, 32),
            ELU(),
            Linear(32, 32),
        )

        self.local_nn_2 = torch.nn.Sequential(
            Linear(35, 35),
            ELU(),
            Linear(35, 35),
            ELU(),
            Linear(35, 35)
        )

        self.global_nn_2 = torch.nn.Sequential(
            Linear(35, 35),
            ELU(),
            Linear(35, 35),
            ELU(),
            Linear(35, 35)
        )

        self.pointnet_conv1 = PointNetConv(self.local_nn_1, self.global_nn_1)

        self.top_k_pooling = TopKPooling(32, ratio=0.2)

        self.pointnet_conv2 = PointNetConv(self.local_nn_2, self.global_nn_2)

        self.aggr = SortAggregation(1)
        self.lin1 = Linear(35, 2)

    def forward(self, data):
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        x_1 = self.pointnet_conv1(x, pos, edge_index)
        x_1 = F.elu(x_1)

        x_1_pool_x, x_1_pool_edge_index, x_1_pool_edge_attr, x_1_pool_batch, x_1_perm, x_1_score = self.top_k_pooling(
            x_1, edge_index, batch=batch)

        x_2 = self.pointnet_conv2(x_1, pos, edge_index)
        x_2 = F.elu(x_2)
        # pointnet_concat = torch.cat([x_1, x_2], dim=1)
        # x = self.aggr(x_2, batch)
        # x = self.aggr(x_1, batch)
        x_max = global_max_pool(x_2, batch)
        x_mean = global_mean_pool(x_2, batch)
        x_add = global_add_pool(x_2, batch)
        # x_norm_pool = torch.cat([x_max, x_mean, x_add], dim=1)
        # x_norm_pool = self.lin1(x_norm_pool)
        x = self.lin1(x_max)
        return x


class GNN_v7(torch.nn.Module):
    def __init__(self):
        super(GNN_v7, self).__init__()
        # try HeteroData?
        self.jet_mlp = torch_geometric.nn.MLP([4, 64, 128], norm=None)
        self.lepton_mlp = torch_geometric.nn.MLP([3, 64, 128], norm=None)
        self.missing_energy_mlp = torch_geometric.nn.MLP([2, 64, 128], norm=None)
        self.high_level_mlp = torch_geometric.nn.MLP([7, 64, 128], norm=None)

        # self.gat_conv1 = GATv2Conv(128, 128)
        self.gat_conv1 = ARMAConv(128, 128)
        self.gat_conv2 = ARMAConv(128, 128)
        # self.gat_conv2 = GATv2Conv(256, 256)

        self.classifier = torch_geometric.nn.MLP([128, 64, 1], norm=None)
        self.edge_index = erdos_renyi_graph(6, 0.5).to('cuda')

    def forward(self, x):
        lepton_raw = x[:, 0:3]
        missing_energy_raw = x[:, 3:5]
        jet1_raw = x[:, 5:9]
        jet2_raw = x[:, 9:13]
        jet3_raw = x[:, 13:17]
        jet4_raw = x[:, 17:21]
        high_level_raw = x[:, 21:28]

        lepton = self.lepton_mlp(lepton_raw)
        missing_energy = self.missing_energy_mlp(missing_energy_raw)
        jet1 = self.jet_mlp(jet1_raw)
        jet2 = self.jet_mlp(jet2_raw)
        jet3 = self.jet_mlp(jet3_raw)
        jet4 = self.jet_mlp(jet4_raw)
        high_level = self.high_level_mlp(high_level_raw)

        x = torch.stack([lepton, missing_energy, jet1, jet2, jet3, jet4, high_level], dim=1)
        graph = torch_geometric.data.Data(x=x[0], edge_index=self.edge_index)
        graph = graph.to('cuda')

        x = self.gat_conv1(graph.x, graph.edge_index)
        x = self.gat_conv2(x, graph.edge_index)
        x = global_max_pool(x, torch.zeros(x.shape[0], dtype=torch.int64).to('cuda'))

        x = self.classifier(x)
        return x

class GNN_v8(torch.nn.Module):
    # Jet index encoding?
    # An End-to-End Deep Learning Architecture for Graph Classification
    def __init__(self):
        super(GNN_v8, self).__init__()
        out_hidden = 512
        self.gat_conv1 = GATv2Conv(3, 64)
        self.gat_conv1_norm = GraphNorm(64)
        self.gat_conv2 = GATv2Conv(64, 128)
        self.gat_conv2_norm = GraphNorm(128)
        self.gat_conv3 = GATv2Conv(128, out_hidden)
        self.gat_conv3_norm = GraphNorm(out_hidden)

        self.gcn_conv1 = ARMAConv(3, 64)
        self.gcn_conv1_norm = GraphNorm(64)
        self.gcn_conv2 = ARMAConv(64, 128)
        self.gcn_conv2_norm = GraphNorm(128)
        self.gcn_conv3 = ARMAConv(128, out_hidden)
        self.gcn_conv3_norm = GraphNorm(out_hidden)
        self.aggr = SortAggregation(4)
        self.lin1 = Linear(out_hidden * 3 * 2, out_hidden)
        self.lin2 = Linear(out_hidden * 4 * 2, out_hidden)

        self.lin3 = Linear(out_hidden * 2 + 128, 1, bias=False)
        self.lin4 = Linear(9, 128)
        # self.lin4 = Linear(2, 2, bias=False) 

    def forward(self, x, edge_index, batch, additional_feat=None):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = batch.max() + 1
        gat_conv1 = self.gat_conv1(x, edge_index)
        gat_conv1 = F.elu(gat_conv1)
        gat_conv1 = self.gat_conv1_norm(gat_conv1, batch)

        gat_conv2 = self.gat_conv2(gat_conv1, edge_index)
        gat_conv2 = F.elu(gat_conv2)
        gat_conv2 = self.gat_conv2_norm(gat_conv2, batch)

        gat_conv3 = self.gat_conv3(gat_conv2, edge_index)
        gat_conv3 = F.elu(gat_conv3)
        gat_conv3 = self.gat_conv3_norm(gat_conv3, batch)

        gcn_conv1 = self.gcn_conv1(x, edge_index)
        gcn_conv1 = F.elu(gcn_conv1)
        gcn_conv1 = self.gcn_conv1_norm(gcn_conv1, batch)

        gcn_conv2 = self.gcn_conv2(gcn_conv1, edge_index)
        gcn_conv2 = F.elu(gcn_conv2)
        gcn_conv2 = self.gcn_conv2_norm(gcn_conv2, batch)

        gcn_conv3 = self.gcn_conv3(gcn_conv2, edge_index)
        gcn_conv3 = F.elu(gcn_conv3)
        gcn_conv3 = self.gcn_conv3_norm(gcn_conv3, batch)

        gcn_gat = torch.cat([gat_conv3, gcn_conv3], dim=1)

        x_max = global_max_pool(gcn_gat, batch)
        x_mean = global_mean_pool(gcn_gat, batch)
        x_add = global_add_pool(gcn_gat, batch)
        x_norm_pool = torch.cat([x_max, x_mean, x_add], dim=1)
        x_norm_pool = self.lin1(x_norm_pool)

        x_agg_pool = self.aggr(gcn_gat, batch)
        x_agg_pool = self.lin2(x_agg_pool)

        if additional_feat is not None:
            additional_feat = additional_feat.reshape(batch_size, 9)
            additional_feat = self.lin4(additional_feat)
            x = torch.cat([x_norm_pool, x_agg_pool, additional_feat], dim=1)
        else:
            x = torch.cat([x_norm_pool, x_agg_pool], dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        # x = self.lin4(x)
        return x

# try HeteroData?

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
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.lin5(x)
        return x
