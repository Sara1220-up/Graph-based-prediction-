import os
import torch
import glob
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

# ==== PATH SETUP ====
graph_dir = "D:/graph_dataset"
model_path = os.path.join(graph_dir, "gnn_trained_model.pth")

# ==== FIND LATEST GRAPH FILE ====
graph_files = sorted(glob.glob(os.path.join(graph_dir, "graph_snapshot_.pt")), key=os.path.getmtime, reverse=True)
if not graph_files:
    print("❌ No graph snapshot found!")
    exit()

latest_graph = graph_files[0]
print(f"📂 Loading latest graph snapshot: {latest_graph}")

# ==== LOAD GRAPH ====
data = torch.load(latest_graph)

# ==== MODEL DEFINITION ====
class TrafficGNN(nn.Module):
    def __init__(self):
        super(TrafficGNN, self).__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ==== LOAD MODEL ====
model = TrafficGNN()
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model loaded successfully.")

# ==== PREDICT ====
with torch.no_grad():
    predictions = torch.sigmoid(model(data))

print("\n🚦 Predicted congestion scores:")
for i, score in enumerate(predictions):
    label = int(data.y[i].item()) if hasattr(data, 'y') else 'N/A'
    print(f"Node {i}: Score = {score.item():.4f}, Label = {label}")
