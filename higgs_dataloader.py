import torch
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class HiggsDataset(Dataset):
    def __init__(self, csv_file, edge_index, split, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load data
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
        # Extract data and label
        data = self.higgs_frame.iloc[idx, 1:].values
        data = data.astype('float').reshape(28, -1)  # Adjust the shape according to your needs
        class_label = self.higgs_frame.iloc[idx, 0].astype('float')

        sample = Data(x=torch.tensor(data, dtype=torch.float), edge_index=self.edge_index,
                      y=torch.tensor(class_label, dtype=torch.long))

        return sample


if __name__ == '__main__':
    # Example of usage and testing
    csv_file = 'data/HIGGS.csv.gz'
    edge_index_ba = barabasi_albert_graph(28, 14)
    higgs_dataset_train = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='train')
    higgs_dataset_val = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='val')
    higgs_dataset_test = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='test')

    print('Length of datasets')
    print(len(higgs_dataset_train))
    print(len(higgs_dataset_val))
    print(len(higgs_dataset_test))

    print('Dataset num_features' + str(higgs_dataset_train.num_features))
    print('Dataset num_classes' + str(higgs_dataset_train.num_classes))

    print('First item of train dataset')
    first_item = higgs_dataset_train[0]
    print(first_item)

    print('Data loader')
    loader = DataLoader(higgs_dataset_train, batch_size=256)
    for data in loader:
        print(data)
        print(data.num_graphs)
        break