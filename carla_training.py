import carla
import time
import networkx as nx
import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# ==== SETUP SAVE DIRECTORY ====
save_dir = os.path.abspath("D:/graph_dataset")
os.makedirs(save_dir, exist_ok=True)
timestamp = int(time.time())

# ==== CONNECT TO CARLA ====
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# ==== CLEANUP EXISTING VEHICLES ====
existing_vehicles = world.get_actors().filter('vehicle.*')
for vehicle in existing_vehicles:
    vehicle.destroy()
print("\u267b\ufe0f Cleared existing vehicles from the world.")

# ==== SPAWN VEHICLES ====
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicles = []

for i in range(15):
    if i >= len(spawn_points):
        break
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
    if vehicle:
        vehicle.set_autopilot(True)
        vehicles.append(vehicle)
        print(f"✅ Vehicle {vehicle.id} spawned at spawn point {i}")
    else:
        print(f"❌ Vehicle NOT spawned at spawn point {i}")

if not vehicles:
    print("❌ No vehicles were spawned. Exiting.")
    exit()

print("⌛ Vehicles are moving for 10 seconds...")
time.sleep(10)

# ==== BUILD NETWORKX GRAPH ====
graph = nx.DiGraph()

for vehicle in vehicles:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    node_id = f"vehicle_{vehicle.id}"
    x, y = transform.location.x, transform.location.y
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    graph.add_node(node_id, x=x, y=y, speed=speed)

# ==== Add Edges Based on Proximity ====
nodes = list(graph.nodes(data=True))
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        n1, d1 = nodes[i]
        n2, d2 = nodes[j]
        dist = math.hypot(d1['x'] - d2['x'], d1['y'] - d2['y'])
        if dist < 50:
            graph.add_edge(n1, n2, weight=dist)

# ==== ADD ADVANCED NODE FEATURES ====
for node in graph.nodes():
    neighbors = list(graph.neighbors(node))
    neighbor_count = len(neighbors)
    avg_edge_distance = (
        sum(graph[node][nbr]['weight'] for nbr in neighbors) / neighbor_count
        if neighbor_count > 0 else 0.0
    )

    x = graph.nodes[node]['x']
    y = graph.nodes[node]['y']
    speed = graph.nodes[node]['speed']

    graph.nodes[node]['x'] = torch.tensor([x, y, speed, neighbor_count, avg_edge_distance], dtype=torch.float)

# ==== CONVERT TO PyG DATA ====
pyg_data = from_networkx(graph)
pyg_data.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes()])

# ==== CREATE DUMMY LABELS (congestion if speed < 2.0) ====
labels = [1 if graph.nodes[node]['x'][2] < 2.0 else 0 for node in graph.nodes()]
pyg_data.y = torch.tensor(labels, dtype=torch.float).view(-1, 1)

print("\n✅ PyG Graph Created!")
print("Node features shape:", pyg_data.x.shape)
print("Edge index shape:", pyg_data.edge_index.shape)

# ==== SAVE THE GRAPH SNAPSHOT ====
graph_path = os.path.join(save_dir, f"graph_snapshot_{timestamp}.pt")
torch.save(pyg_data, graph_path)
print(f"📦 Graph snapshot saved to {graph_path}")

# ==== DEFINE & TRAIN GNN MODEL ====
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

model = TrafficGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

print("\n📚 Training GNN for 50 epochs...")
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(pyg_data)
    loss = loss_fn(out, pyg_data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 49:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# ==== INFERENCE ====
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(pyg_data))

print("\n🚦 Predicted congestion scores per vehicle:")
for i, node in enumerate(graph.nodes()):
    score = predictions[i].item()
    label = int(pyg_data.y[i].item())
    print(f"{node}: Score = {score:.4f}, Label = {label}")

# ==== SAVE TRAINED MODEL ====
model_path = os.path.join(save_dir, "gnn_trained_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Trained model saved to: {model_path}")
# ==== CLEANUP ====
print("\n⌛ Waiting 30 seconds to view visualization...")
time.sleep(30)
for v in vehicles:
    v.destroy()
print("\n✅ Vehicles destroyed. Visualization complete.")


