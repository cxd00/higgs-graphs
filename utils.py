import random

import numpy as np
from torch_geometric.utils import to_networkx, erdos_renyi_graph
import networkx as nx
import torch
from visualize import GraphVisualization

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20, node_text=[str(i) for i in graph.node_name]
    )
    fig = vis.create_figure()
    return fig

def generate_higgs_exp_graph_edge():
    # Generate a fully connect graph between edge 0 and 5
    fully_connect_physical = erdos_renyi_graph(5, 1.0)
    fully_connect_jet_1 = erdos_renyi_graph(4, 1.0) + 5
    fully_connect_jet_2 = erdos_renyi_graph(4, 1.0) + 9
    fully_connect_jet_3 = erdos_renyi_graph(4, 1.0) + 13
    fully_connect_jet_4 = erdos_renyi_graph(4, 1.0) + 17
    fully_connect_manual_feat = erdos_renyi_graph(7, 1.0) + 21
    connectionn = torch.tensor([[5, 9, 13, 17, 21], [0, 0, 0, 0, 0]])
    all_edge = torch.cat(
        [fully_connect_physical, fully_connect_jet_1, fully_connect_jet_2, fully_connect_jet_3, fully_connect_jet_4,
         fully_connect_manual_feat, connectionn], dim=1)
    return all_edge

def generate_higgs_exp_graph_edge_v3():
    # Generate a fully connect graph between edge 0 and 5
    fully_connect_physical = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]])
    fully_connect_jet_1 = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]) + 5
    fully_connect_jet_2 = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]) + 9
    fully_connect_jet_3 = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]) + 13
    fully_connect_jet_4 = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]) + 17
    fully_connect_manual_feat = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]]) + 21
    connection = torch.tensor([[5, 9, 13, 17, 21], [0, 1, 2, 3, 4]])
    all_edge = torch.cat(
        [fully_connect_physical, fully_connect_jet_1, fully_connect_jet_2, fully_connect_jet_3, fully_connect_jet_4,
         fully_connect_manual_feat, connection], dim=1)
    return all_edge

def generate_higgs_exp_graph_edge_v4():
    fully_connect_physical = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 2, 1, 2,6]])
    return fully_connect_physical

def generate_higgs_exp_graph_edge_v5():
    fully_connect_physical = torch.tensor([[0, 0, 1, 2, 3, 4, 7], [5, 1, 6, 1, 2, 2, 5]])
    return fully_connect_physical

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False