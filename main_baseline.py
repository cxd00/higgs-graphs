import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import barabasi_albert_graph

from model import GNN,MLP
from higgs_dataloader import HiggsDatasetTorch
import torch.nn.functional as F

csv_file = 'data/HIGGS.csv.gz'
edge_index_ba = barabasi_albert_graph(28, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GNN().to(device)
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Test PyG
# Test PyTorch
train_dataset = HiggsDatasetTorch(csv_file=csv_file, split='train')
val_dataset = HiggsDatasetTorch(csv_file=csv_file, split='val')
test_dataset = HiggsDatasetTorch(csv_file=csv_file, split='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)


# Set the number of epochs
num_epochs = 10

# Start the training loop
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out, label.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Eval
    val_loss = 0
    model.eval()
    for data, label in val_loader:
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out, label.reshape(-1, 1))
        val_loss += loss.item()

    print(f'Epoch {epoch}, Train loss: {train_loss/len(train_loader)}, Val loss: {val_loss/len(val_loader)}')

# Test with roc_auc_score
model.eval()
preds = []
labels = []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        out = model(data)
        preds.append(out.detach().cpu())
        labels.append(label.detach().cpu())

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
