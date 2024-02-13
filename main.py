import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.loader import DataLoader

from model import GNN
from higgs_dataloader import HiggsDataset
import torch.nn.functional as F

csv_file = 'data/HIGGS.csv.gz'
edge_index_ba = barabasi_albert_graph(28, 14)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='train')
val_dataset = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='val')
test_dataset = HiggsDataset(csv_file=csv_file, edge_index=edge_index_ba, split='test')

train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

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

