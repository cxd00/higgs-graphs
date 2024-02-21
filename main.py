import plotly
import torch
import torch_geometric
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import barabasi_albert_graph, erdos_renyi_graph
from torch_geometric.nn import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import GNN, MLP, GNN_v2, GNN_v3, GNN_v3_mini, GNN_v4, GNN_v5
from higgs_dataloader import HiggsDatasetPyG, HiggsDatasetTorch
import torch.nn.functional as F

from utils import generate_higgs_exp_graph_edge, create_graph, set_seed, generate_higgs_exp_graph_edge_v3

# TODO add wandb support
# https://colab.research.google.com/github/wandb/examples/blob/pyg/graph-classification/colabs/pyg/Graph_Classification_with_PyG_and_W%26B.ipynb#scrollTo=elAN_YlM_Pyr

wandb.init(project='higgs-graphs')
set_seed(0)

csv_file = 'data/HIGGS.csv.gz'
# edge_index_ba = barabasi_albert_graph(28, 8)
# edge_index_ba = erdos_renyi_graph(28, 1.0)
edge_index_ba = generate_higgs_exp_graph_edge()
# edge_index_ba = generate_higgs_exp_graph_edge_v3()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GNN()
# model = GNN_v2()
# model = GNN_v3()
# model = GNN_v3_mini()
# model = GNN_v4()
model = GNN_v5()
# model = MLP()

# wandb log model name
wandb.config.model_name = model.__class__.__name__

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Test PyG
print('Loading datasets')
train_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='train')
val_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='val')
test_dataset = HiggsDatasetPyG(csv_file=csv_file, edge_index=edge_index_ba, split='test')

# upload graph vis to wandb
table = wandb.Table(columns=["Graph", "Label"])
fig = create_graph(train_dataset[0])
label = train_dataset[0].y.argmax()
table.add_data(wandb.Html(plotly.io.to_html(fig)), label)
wandb.log({"data": table})

train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=1)

# Set the number of epochs
num_epochs = 35
print(summary(model, train_dataset[0]))
# writer = SummaryWriter()
# writer.add_graph(model, [train_dataset[0].x, train_dataset[0].edge_index, torch.tensor([0])])

# Start the training loop
model = model.to(device)
# model = torch.compile(model)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print('Start training')
try:
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
            # loss = F.mse_loss(out, data.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        # Eval
        val_loss = 0
        model.eval()
        for data in tqdm(val_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
            val_loss += loss.item()

        print(f'Epoch {epoch}, Train loss: {train_loss / len(train_loader)}, Val loss: {val_loss / len(val_loader)}, '
              f'learning rate: {optimizer.param_groups[0]["lr"]}')
        wandb.log({"train_loss": train_loss / len(train_loader), "val_loss": val_loss / len(val_loader),
                   "learning_rate": optimizer.param_groups[0]["lr"]})
except KeyboardInterrupt:
    print('Training terminated')

# Test with roc_auc_score
model.eval()
preds = []
labels = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Test set evaluation"):
        data = data.to(device)
        out = model(data)
        preds.append(out.detach().cpu())
        labels.append(data.y.detach().cpu())

# Convert lists to tensors
preds = torch.cat(preds).numpy()
labels = torch.cat(labels).numpy()

# Compute ROC AUC score
roc_auc = roc_auc_score(labels, preds)
print(f'Test ROC AUC score: {roc_auc}')
wandb.log({"roc_auc": roc_auc})
