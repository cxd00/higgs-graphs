import math

import torch
import pandas as pd
import torch_geometric.data
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import barabasi_albert_graph, erdos_renyi_graph
from torch_geometric.data import Data

from utils import create_graph, generate_higgs_exp_graph_edge


class HiggsDatasetPyG(torch_geometric.data.Dataset):
    def __init__(self, csv_file, edge_index, split, norm=False, drop_feats=False, root=None, transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load data
        self.higgs_frame = pd.read_csv(csv_file, compression='gzip', header=None, nrows=20000)
        # enable shuffle
        self.higgs_frame = self.higgs_frame.sample(frac=1).reset_index(drop=True)
        self.higgs_frame.columns = [
            'class_label',
            'lepton_pT', 'lepton_eta', 'lepton_phi',
            'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
            'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
            'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
            'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        if drop_feats:
            self.higgs_frame = self.higgs_frame.drop(
                columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
        if norm:
            self.higgs_frame = self.normalize(self.higgs_frame)
        self.edge_index = edge_index
        total_rows = len(self.higgs_frame)
        if split == 'train':
            self.higgs_frame = self.higgs_frame[:int(total_rows * 0.6)]
        elif split == 'val':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.6):int(total_rows * 0.8)]
        elif split == 'test':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.8):]

    def len(self):
        return len(self.higgs_frame)

    def get(self, idx):
        data = self.higgs_frame.iloc[idx, 1:].values
        data_len = len(data)
        data = data.astype('float').reshape(data_len, -1)
        class_label = self.higgs_frame.iloc[idx, 0]
        # class_label_one_hot = torch.zeros(2)
        # class_label_one_hot[int(class_label)] = 1

        # edge_attr = torch.ones((self.edge_index.shape[1], 1))

        sample = Data(x=torch.tensor(data, dtype=torch.float), edge_index=self.edge_index,
                      y=torch.tensor(int(class_label), dtype=torch.float),
                      node_name=self.higgs_frame.columns[1:].values)

        return sample

    def normalize(self, df):
        columns_to_normalize_0_1 = ['jet_1_b-tag', 'jet_2_b-tag', 'jet_3_b-tag', 'jet_4_b-tag',
                                    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

        columns_to_normalize_minus1_1 = [col for col in df.columns if
                                         col not in columns_to_normalize_0_1 + ['class_label']]
        data_0_1 = df[columns_to_normalize_0_1]
        data_minus1_1 = df[columns_to_normalize_minus1_1]
        scaler_0_1 = MinMaxScaler(feature_range=(0, 1))
        data_0_1 = pd.DataFrame(scaler_0_1.fit_transform(data_0_1), columns=data_0_1.columns)
        scaler_minus1_1 = MinMaxScaler(feature_range=(-1, 1))
        data_minus1_1 = pd.DataFrame(scaler_minus1_1.fit_transform(data_minus1_1), columns=data_minus1_1.columns)

        normalized_df = pd.concat([df['class_label'], data_0_1, data_minus1_1], axis=1)
        normalized_df = normalized_df.reindex(columns=df.columns)

        return normalized_df

class HiggsDatasetNewPyG(torch_geometric.data.Dataset):
    def __init__(self, csv_file, edge_index, split, norm=False, drop_feats=False, root=None, transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load data
        self.higgs_frame = pd.read_csv(csv_file, compression='gzip', header=None, nrows=20000)
        # enable shuffle
        self.higgs_frame = self.higgs_frame.sample(frac=1).reset_index(drop=True)
        self.higgs_frame.columns = [
            'class_label',
            'lepton_pT', 'lepton_eta', 'lepton_phi',
            'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
            'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
            'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
            'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        if drop_feats:
            self.higgs_frame = self.higgs_frame.drop(
                columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
        if norm:
            self.higgs_frame = self.normalize(self.higgs_frame)
        self.edge_index = edge_index
        total_rows = len(self.higgs_frame)
        if split == 'train':
            self.higgs_frame = self.higgs_frame[:int(total_rows * 0.6)]
        elif split == 'val':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.6):int(total_rows * 0.8)]
        elif split == 'test':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.8):]

    def len(self):
        return len(self.higgs_frame)

    def get(self, idx):
        data = self.higgs_frame.iloc[idx, 1:].values
        data_len = len(data)
        data = data.astype('float')
        class_label = self.higgs_frame.iloc[idx, 0]

        lepton_raw = data[0:3]
        missing_energy_raw = data[3:5]
        jet1_raw = data[5:9]
        jet2_raw = data[9:13]
        jet3_raw = data[13:17]
        jet4_raw = data[17:21]
        high_level_raw = data[21:28]

        # sort jet1-4 by last elemnt of each jet_raw
        jet_total = torch.tensor([jet1_raw, jet2_raw, jet3_raw, jet4_raw])
        jet_total = jet_total[jet_total[ :, 3].argsort()]
        
        w_jet_1 = jet_total[0, :3]
        w_jet_2 = jet_total[1, :3]
        b_jet_1 = jet_total[2, :3]
        b_jet_2 = jet_total[3, :3]

        empty_node_1 = torch.zeros(3)
        empty_node_2 = torch.zeros(3)
        empty_node_3 = torch.zeros(3)

        lepton_raw = torch.tensor(lepton_raw, dtype=torch.float)
        missing_energy_raw = torch.tensor(missing_energy_raw, dtype=torch.float)
        high_level_raw = torch.tensor(high_level_raw, dtype=torch.float)

        # node
        node_names = ['empty1', 'empty2', 'empty3', 'b_jet_1', 'b_jet_2', 'w_jet_1', 'w_jet_2', 'lepton']
        data = torch.stack([empty_node_1, empty_node_2, empty_node_3, b_jet_1, b_jet_2, w_jet_1, w_jet_2, lepton_raw])

        sample = Data(x=torch.tensor(data, dtype=torch.float), edge_index=self.edge_index,
                      y=torch.tensor(int(class_label), dtype=torch.float),
                      node_name=node_names)
        sample.additional_feat = torch.concat([missing_energy_raw, high_level_raw], dim=0)

        return sample

    def normalize(self, df):
        columns_to_normalize_0_1 = ['jet_1_b-tag', 'jet_2_b-tag', 'jet_3_b-tag', 'jet_4_b-tag',
                                    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

        columns_to_normalize_minus1_1 = [col for col in df.columns if
                                         col not in columns_to_normalize_0_1 + ['class_label']]
        data_0_1 = df[columns_to_normalize_0_1]
        data_minus1_1 = df[columns_to_normalize_minus1_1]
        scaler_0_1 = MinMaxScaler(feature_range=(0, 1))
        data_0_1 = pd.DataFrame(scaler_0_1.fit_transform(data_0_1), columns=data_0_1.columns)
        scaler_minus1_1 = MinMaxScaler(feature_range=(-1, 1))
        data_minus1_1 = pd.DataFrame(scaler_minus1_1.fit_transform(data_minus1_1), columns=data_minus1_1.columns)

        normalized_df = pd.concat([df['class_label'], data_0_1, data_minus1_1], axis=1)
        normalized_df = normalized_df.reindex(columns=df.columns)

        return normalized_df

class HiggsDataset3DPyG(torch_geometric.data.Dataset):
    def __init__(self, csv_file, split, norm=False, drop_feats=True, root=None, transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load data
        self.higgs_frame = pd.read_csv(csv_file, compression='gzip', header=None, nrows=20000)
        # enable shuffle
        self.higgs_frame = self.higgs_frame.sample(frac=1).reset_index(drop=True)
        self.higgs_frame.columns = [
            'class_label',
            'lepton_pT', 'lepton_eta', 'lepton_phi',
            'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
            'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
            'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
            'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        self.drop_feats = drop_feats
        if norm:
            self.higgs_frame = self.normalize(self.higgs_frame)
        self.edge_index = self.calculate_edge_index()
        total_rows = len(self.higgs_frame)
        if split == 'train':
            self.higgs_frame = self.higgs_frame[:int(total_rows * 0.6)]
        elif split == 'val':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.6):int(total_rows * 0.8)]
        elif split == 'test':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.8):]

    def len(self):
        return len(self.higgs_frame)

    def get(self, idx):
        # data = self.higgs_frame.iloc[idx, 1:].values
        # data_len = len(data)
        # data = data.astype('float').reshape(data_len, -1)
        # class_label = self.higgs_frame.iloc[idx, 0]
        # class_label_one_hot = torch.zeros(2)
        # class_label_one_hot[int(class_label)] = 1
        #
        # sample = Data(x=torch.tensor(data, dtype=torch.float), edge_index=self.edge_index,
        #               y=torch.tensor(torch.unsqueeze(class_label_one_hot,0), dtype=torch.float), node_name=self.higgs_frame.columns[1:].values)

        row = self.higgs_frame.iloc[idx]
        if self.drop_feats:
            node_0 = [0, 0, 0]
        else:
            node_0 = row[['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']].values
        jets_vector, jets_feature = self.generate_3d_jet(self.higgs_frame.iloc[idx])

        x = [node_0] + jets_feature
        x_pos = [[0, 0, 0]] + jets_vector
        x = torch.tensor(x, dtype=torch.float)
        x_pos = torch.tensor(x_pos, dtype=torch.float)
        class_label_one_hot = torch.zeros(2)
        class_label_one_hot[int(row['class_label'])] = 1.0
        y = torch.unsqueeze(class_label_one_hot, 0)
        sample = Data(x=x, pos=x_pos, edge_index=self.edge_index, y=y,
                      node_name=['center', 'jet_1', 'jet_2', 'jet_3', 'jet_4', 'lepton', 'missing_energy'], num_nodes=7)
        return sample

    def generate_3d_jet(self, row):
        jet_1 = self.process_single_jet(row['jet_1_pt'], row['jet_1_eta'], row['jet_1_phi'])
        jet_1_feature = [row['jet_1_b-tag'], 0, 0]
        jet_2 = self.process_single_jet(row['jet_2_pt'], row['jet_2_eta'], row['jet_2_phi'])
        jet_2_feature = [row['jet_2_b-tag'], 0, 0]
        jet_3 = self.process_single_jet(row['jet_3_pt'], row['jet_3_eta'], row['jet_3_phi'])
        jet_3_feature = [row['jet_3_b-tag'], 0, 0]
        jet_4 = self.process_single_jet(row['jet_4_pt'], row['jet_4_eta'], row['jet_4_phi'])
        jet_4_feature = [row['jet_4_b-tag'], 0, 0]
        lepton_jet = self.process_single_jet(row['lepton_pT'], row['lepton_eta'], row['lepton_phi'])
        lepton_jet_feature = [0, 1, 0]
        missing_energy_jet = self.process_single_jet(row['missing_energy_magnitude'], 0, row['missing_energy_phi'])
        missing_energy_jet_feature = [0, 0, 1]

        return [jet_1, jet_2, jet_3, jet_4, lepton_jet, missing_energy_jet], [jet_1_feature, jet_2_feature,
                                                                              jet_3_feature, jet_4_feature,
                                                                              lepton_jet_feature,
                                                                              missing_energy_jet_feature]

    def process_single_jet(self, jet_pt, jet_eta, jet_phi):
        jet_theta = 2 * math.atan(math.exp(-jet_eta))
        x = jet_pt * math.sin(jet_theta) * math.cos(jet_phi)
        y = jet_pt * math.sin(jet_theta) * math.sin(jet_phi)
        z = jet_pt * math.cos(jet_theta)
        return [x, y, z]

    def normalize(self, df):
        columns_to_normalize = ['jet_1_b-tag', 'jet_2_b-tag', 'jet_3_b-tag', 'jet_4_b-tag',
                                'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        for column in columns_to_normalize:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    def calculate_edge_index(self):
        # replace with some edge calculation?
        return torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]])


class HiggsDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, csv_file, split, norm=False, drop_feats=False):
        super(HiggsDatasetTorch, self).__init__()
        self.higgs_frame = pd.read_csv(csv_file, compression='gzip', header=None, nrows=10000)
        self.higgs_frame.columns = [
            'class_label',
            'lepton_pT', 'lepton_eta', 'lepton_phi',
            'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
            'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
            'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
            'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        if drop_feats:
            self.higgs_frame = self.higgs_frame.drop(
                columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
        if norm:
            self.higgs_frame = self.normalize(self.higgs_frame)
        total_rows = len(self.higgs_frame)
        if split == 'train':
            self.higgs_frame = self.higgs_frame[:int(total_rows * 0.6)]
        elif split == 'val':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.6):int(total_rows * 0.8)]
        elif split == 'test':
            self.higgs_frame = self.higgs_frame[int(total_rows * 0.8):]

    def __len__(self):
        return len(self.higgs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.higgs_frame.iloc[idx, 1:].values.astype('float32')
        label = self.higgs_frame.iloc[idx, 0].astype('float32')

        return torch.tensor(features, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, df):
        columns_to_normalize_0_1 = ['jet_1_b-tag', 'jet_2_b-tag', 'jet_3_b-tag', 'jet_4_b-tag',
                                    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

        columns_to_normalize_minus1_1 = [col for col in df.columns if
                                         col not in columns_to_normalize_0_1 + ['class_label']]
        data_0_1 = df[columns_to_normalize_0_1]
        data_minus1_1 = df[columns_to_normalize_minus1_1]
        scaler_0_1 = MinMaxScaler(feature_range=(0, 1))
        data_0_1 = pd.DataFrame(scaler_0_1.fit_transform(data_0_1), columns=data_0_1.columns)
        scaler_minus1_1 = MinMaxScaler(feature_range=(-1, 1))
        data_minus1_1 = pd.DataFrame(scaler_minus1_1.fit_transform(data_minus1_1), columns=data_minus1_1.columns)

        normalized_df = pd.concat([df['class_label'], data_0_1, data_minus1_1], axis=1)
        normalized_df = normalized_df.reindex(columns=df.columns)

        return normalized_df

if __name__ == '__main__':
    # Example of usage and testing
    csv_file = 'data/HIGGS.csv.gz'

    edge_index_hg = generate_higgs_exp_graph_edge()

    # higgs_dataset_train = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_hg, split='train')
    # higgs_dataset_val = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_hg, split='val')
    # higgs_dataset_test = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_hg, split='test')
    # higgs_dataset_train = HiggsDataset3DPyG(csv_file=csv_file, split='train')
    # higgs_dataset_val = HiggsDataset3DPyG(csv_file=csv_file, split='val')
    # higgs_dataset_test = HiggsDataset3DPyG(csv_file=csv_file, split='test')
    higgs_dataset_train = HiggsDatasetNewPyG(csv_file=csv_file, edge_index=edge_index_hg, split='train')
    higgs_dataset_val = HiggsDatasetNewPyG(csv_file=csv_file, edge_index=edge_index_hg, split='val')
    higgs_dataset_test = HiggsDatasetNewPyG(csv_file=csv_file, edge_index=edge_index_hg, split='test')

    print('Length of datasets')
    print(len(higgs_dataset_train))
    print(len(higgs_dataset_val))
    print(len(higgs_dataset_test))

    print('Dataset num_features' + str(higgs_dataset_train.num_features))
    print('Dataset num_classes' + str(higgs_dataset_train.num_classes))

    print('First item of train dataset')
    first_item = higgs_dataset_train[0]
    print(first_item)
    print('number of nodes' + str(first_item.num_nodes))
    print('number of edges' + str(first_item.num_edges))
    print('number of node features' + str(first_item.num_node_features))
    print('has_isolated_nodes' + str(first_item.has_isolated_nodes()))
    print('contains_self_loops' + str(first_item.contains_self_loops()))
    # print('edge_attr.shape' + str(first_item.edge_attr.shape))
    print('is_directed' + str(first_item.is_directed()))
    fig = create_graph(first_item)
    fig.show()

    print('Data loader')
    loader = torch_geometric.loader.DataLoader(higgs_dataset_train, batch_size=256)
    for data in loader:
        print(data)
        print(data.num_graphs)
        break

    print('Testing pytorch dataset')
    higgs_dataset_torch = HiggsDatasetTorch(csv_file=csv_file, split='train')
    print(len(higgs_dataset_torch))
    print(higgs_dataset_torch[0])
