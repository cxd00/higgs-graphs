import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import barabasi_albert_graph

from model import GNN,MLP
from higgs_dataloader import HiggsDatasetPyG, HiggsDatasetTorch
import torch.nn.functional as F

csv_file = 'data/HIGGS.csv.gz'
edge_index_ba = barabasi_albert_graph(28, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
# model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Test PyG
train_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='train')
val_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='val')
test_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='test')

train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=64)
val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=64)
test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=64)

# Set the number of epochs
num_epochs = 100

# Start the training loop
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.binary_cross_entropy(out, data.y.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Eval
    val_loss = 0
    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.binary_cross_entropy(out, data.y.reshape(-1, 1))
        val_loss += loss.item()

    print(f'Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}')

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
