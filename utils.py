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
    fully_connect_physical = erdos_renyi_graph(5, 0.5)
    fully_connect_jet_1 = erdos_renyi_graph(4, 0.75) + 5
    fully_connect_jet_2 = erdos_renyi_graph(4, 0.75) + 9
    fully_connect_jet_3 = erdos_renyi_graph(4, 0.75) + 13
    fully_connect_jet_4 = erdos_renyi_graph(4, 0.75) + 17
    fully_connect_manual_feat = erdos_renyi_graph(7, 0.5) + 21
    connectionn = torch.tensor([[5, 9, 13, 17, 21], [0, 0, 0, 0, 0]])
    all_edge = torch.cat([fully_connect_physical, fully_connect_jet_1, fully_connect_jet_2, fully_connect_jet_3, fully_connect_jet_4, fully_connect_manual_feat, connectionn], dim=1)
    return all_edge
