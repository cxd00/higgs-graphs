import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import barabasi_albert_graph, erdos_renyi_graph
from torch_geometric.nn import summary
from torch.utils.tensorboard import SummaryWriter

from model import GNN,MLP
from higgs_dataloader import HiggsDatasetPyG, HiggsDatasetTorch
import torch.nn.functional as F

# TODO add wandb support
#  https://colab.research.google.com/github/wandb/examples/blob/pyg/graph-classification/colabs/pyg/Graph_Classification_with_PyG_and_W%26B.ipynb#scrollTo=elAN_YlM_Pyr

csv_file = 'data/HIGGS.csv.gz'
# edge_index_ba = barabasi_albert_graph(28, 8)
edge_index_ba = erdos_renyi_graph(20, 1.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN()
# model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Test PyG
train_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='train')
val_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='val')
test_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='test')

train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=32)
val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=32)
test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=32)

# Set the number of epochs
num_epochs = 100
print(summary(model, train_dataset[0].x, train_dataset[0].edge_index, torch.tensor([0])))
# writer = SummaryWriter()
# writer.add_graph(model, [train_dataset[0].x, train_dataset[0].edge_index, torch.tensor([0])])

# Start the training loop
model = model.to(device)
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Eval
    val_loss = 0
    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        val_loss += loss.item()

    print(f'Epoch {epoch}, Train loss: {train_loss/len(train_loader)}, Val loss: {val_loss/len(val_loader)}')

# Test with roc_auc_score
model.eval()
preds = []
labels = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.detach().cpu())
        labels.append(data.y.detach().cpu())

# Convert lists to tensors
preds = torch.cat(preds).numpy()
labels = torch.cat(labels).numpy()

# Initialize OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the labels to one-hot encoding
labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))

# Convert to numpy array
labels_one_hot = labels_one_hot.toarray()

# Compute ROC AUC score
roc_auc = roc_auc_score(labels_one_hot, preds)
print(f'Test ROC AUC score: {roc_auc}')
