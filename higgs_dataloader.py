import torch
import pandas as pd
import torch_geometric.data
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.data import Data

class HiggsDatasetPyG(torch_geometric.data.Dataset):
    def __init__(self, csv_file, edge_index, split, norm=False, drop_feats=False, root=None, transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load data
        self.higgs_frame = pd.read_csv(csv_file, compression='gzip', header=None, nrows=10000)
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
            self.higgs_frame = self.higgs_frame.drop(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
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
        class_label = self.higgs_frame.iloc[idx, 0].astype('float')

        edge_attr = torch.ones((self.edge_index.shape[1], 1))

        sample = Data(x=torch.tensor(data, dtype=torch.float), edge_index=self.edge_index, edge_attr=edge_attr,
                      y=torch.tensor([class_label], dtype=torch.long))

        return sample

    def normalize(self, df):
        return (df - df.min()) / (df.max() - df.min())


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
            self.higgs_frame = self.higgs_frame.drop(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
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
        return (df - df.min()) / (df.max() - df.min())


if __name__ == '__main__':
    # Example of usage and testing
    csv_file = 'data/HIGGS.csv.gz'
    edge_index_ba = barabasi_albert_graph(28, 14)
    higgs_dataset_train = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='train')
    higgs_dataset_val = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='val')
    higgs_dataset_test = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='test')

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
    print('edge_attr.shape' + str(first_item.edge_attr.shape))
    print('is_directed' + str(first_item.is_directed()))

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

