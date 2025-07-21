

import carla
import time
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

# === GNN Model ===
class TrafficGNN(torch.nn.Module):
    def __init__(self):
        super(TrafficGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# === Connect to CARLA ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# === Clear world ===
for v in world.get_actors().filter('vehicle.*'):
    v.destroy()
print("♻️ Cleared existing vehicles from the world.")

# === Spawn vehicles ===
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicles = []
for i in range(15):
    if i < len(spawn_points):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
        if vehicle:
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"✅ Vehicle {vehicle.id} spawned")

time.sleep(6)  # Let vehicles move a bit

# === Build graph ===
graph = nx.Graph()
vehicle_lookup = {f"vehicle_{v.id}": v for v in vehicles}

for v in vehicles:
    loc = v.get_transform().location
    vel = v.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    graph.add_node(f"vehicle_{v.id}", x=loc.x, y=loc.y, speed=speed)

nodes = list(graph.nodes(data=True))
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        n1, d1 = nodes[i]
        n2, d2 = nodes[j]
        dist = math.hypot(d1['x'] - d2['x'], d1['y'] - d2['y'])
        if dist < 50:
            graph.add_edge(n1, n2, weight=dist)
            # Draw yellow edge in CARLA
            v1 = vehicle_lookup.get(n1)
            v2 = vehicle_lookup.get(n2)
            if v1 and v2:
                world.debug.draw_line(
                    v1.get_transform().location,
                    v2.get_transform().location,
                    thickness=0.2,
                    color=carla.Color(255, 255, 0),
                    life_time=60
                )

print(f"🔗 Total edges in graph: {graph.number_of_edges()}")

# === Convert to PyG ===
for node in graph.nodes(data=True):
    graph.nodes[node[0]]['x'] = torch.tensor([
        node[1]['x'], node[1]['y'], node[1]['speed']
    ], dtype=torch.float)

data = from_networkx(graph)
data.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes()])

# === Load model + predict ===
model = TrafficGNN()
model.load_state_dict(torch.load("D:/graph_dataset/gnn_trained_model.pth"))
model.eval()

with torch.no_grad():
    predictions = torch.sigmoid(model(data)).squeeze()

print("\n🚦 Predicted Congestion Scores:")
for i, node in enumerate(graph.nodes()):
    score = predictions[i].item()
    print(f"{node}: Score = {score:.4f}")
    graph.nodes[node]['score'] = score

# === Rank top 3 ===
top3 = sorted(
    [(node, graph.nodes[node]['score']) for node in graph.nodes()],
    key=lambda x: x[1],
    reverse=True
)[:3]

print("\n🏆 Top 3 Congested Vehicles:")
for i, (node, score) in enumerate(top3, 1):
    print(f"{i}. {node} — Score: {score:.4f}")

# === Draw in CARLA ===
for node in graph.nodes():
    vehicle = vehicle_lookup.get(node)
    if vehicle:
        loc = vehicle.get_transform().location
        score = graph.nodes[node]['score']
        is_top = node in [t[0] for t in top3]

        label = f"{node}\n{score:.2f}"
        color = carla.Color(255, 0, 0) if is_top else carla.Color(0, 255, 0)

        # Floating label
        world.debug.draw_string(
            location=loc + carla.Location(z=2.0),
            text=label,
            life_time=60.0,
            color=color,
            persistent_lines=True
        )

        # Proper bounding box
        bbox = vehicle.bounding_box
        bbox.location += loc
        world.debug.draw_box(
            bbox,
            rotation=vehicle.get_transform().rotation,
            thickness=0.1,
            color=color,
            life_time=60.0
        )

# === Visualize graph ===
pos = {n: (graph.nodes[n]['x'][0].item(), graph.nodes[n]['x'][1].item()) for n in graph.nodes()}
node_colors = [graph.nodes[n]['score'] for n in graph.nodes()]

plt.figure(figsize=(10, 6))
nx.draw_networkx(
    graph, pos,
    node_color=node_colors,
    cmap=plt.cm.Reds,
    edge_color='black',
    with_labels=True,
    node_size=800,
    font_size=8
)
plt.title("Vehicle Congestion Graph (GNN Prediction)")
plt.legend(
    handles=[
        mpatches.Patch(color='mistyrose', label='Low Congestion'),
        mpatches.Patch(color='darkred', label='High Congestion')
    ],
    loc='upper left'
)
plt.savefig("D:/graph_dataset/predicted_graph.png")
plt.show(block=True)

# === Cleanup ===
print("\n⌛ Waiting 60 seconds before cleanup...")
time.sleep(60)
for v in vehicles:
    v.destroy()
print("✅ Cleanup done. Evaluation complete.")


