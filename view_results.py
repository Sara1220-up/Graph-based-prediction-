import carla
import time
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import csv

# === GNN Model ===
class TrafficGNN(torch.nn.Module):
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

# === Connect to CARLA ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# === Clear world ===
for v in world.get_actors().filter('vehicle.*'):
    v.destroy()
print("[INFO] Cleared existing vehicles from the world.")

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
            print(f"[INFO] Vehicle {vehicle.id} spawned")

# === Load GNN model ===
model = TrafficGNN()
model.load_state_dict(torch.load("D:/graph_dataset/gnn_trained_model.pth"))
model.eval()

# === Run for fixed time ===
start_time = time.time()
last_graph = None
vehicle_lookup = {}
print("[INFO] Running simulation for 30 seconds...")

while time.time() - start_time < 30:
    vehicles = [v for v in vehicles if v.is_alive]
    graph = nx.DiGraph()
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

    for node in graph.nodes(data=True):
        neighbors = list(graph.neighbors(node[0]))
        neighbor_count = len(neighbors)
        avg_edge_distance = (
            sum(graph[node[0]][nbr]['weight'] for nbr in neighbors) / neighbor_count
            if neighbor_count > 0 else 0.0
        )
        graph.nodes[node[0]]['x'] = torch.tensor([
            node[1]['x'], node[1]['y'], node[1]['speed'], neighbor_count, avg_edge_distance
        ], dtype=torch.float)

    data = from_networkx(graph)
    data.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes()])

    with torch.no_grad():
        predictions = torch.sigmoid(model(data)).squeeze()

    for i, node in enumerate(graph.nodes()):
        score = predictions[i].item()
        graph.nodes[node]['score'] = score

    last_graph = graph.copy()

    # === Draw in CARLA ===
    for node in graph.nodes():
        vehicle = vehicle_lookup.get(node)
        if vehicle and vehicle.is_alive:
            tf = vehicle.get_transform()
            loc = tf.location
            score = graph.nodes[node]['score']
            congested = score >= 0.5
            color = carla.Color(255, 0, 0) if congested else carla.Color(0, 255, 0)

            world.debug.draw_string(
                location=loc + carla.Location(z=2.5),
                text=f"{node}\n{score:.2f}",
                life_time=1.0,
                color=color,
                persistent_lines=False
            )

            bbox = vehicle.bounding_box
            bbox_location = loc + bbox.location
            world.debug.draw_box(
                box=carla.BoundingBox(bbox_location, bbox.extent),
                rotation=tf.rotation,
                color=color,
                life_time=1.0,
                persistent_lines=False
            )

    # === Draw edges in CARLA ===
    for edge in graph.edges():
        n1, n2 = edge
        v1 = vehicle_lookup.get(n1)
        v2 = vehicle_lookup.get(n2)
        if v1 and v2 and v1.is_alive and v2.is_alive:
            score1 = graph.nodes[n1]['score']
            score2 = graph.nodes[n2]['score']
            avg_score = (score1 + score2) / 2
            color = carla.Color(255, 0, 0) if avg_score >= 0.5 else carla.Color(0, 255, 0)

            world.debug.draw_line(
                v1.get_transform().location,
                v2.get_transform().location,
                thickness=0.2,
                color=color,
                life_time=1.0,
                persistent_lines=False
            )

    time.sleep(1)

# === SNA Metrics ===
print("[INFO] Simulation complete. Calculating SNA metrics...")
degree = nx.degree_centrality(last_graph)
closeness = nx.closeness_centrality(last_graph)
betweenness = nx.betweenness_centrality(last_graph, k=5, seed=42)  # Approximate
communities = nx.community.label_propagation_communities(last_graph.to_undirected())
comm_dict = {}
for i, c in enumerate(communities):
    for node in c:
        comm_dict[node] = i

for node in last_graph.nodes():
    last_graph.nodes[node]['degree'] = degree[node]
    last_graph.nodes[node]['closeness'] = closeness[node]
    last_graph.nodes[node]['betweenness'] = betweenness[node]
    last_graph.nodes[node]['community'] = comm_dict.get(node, -1)

# === Plot Final Graph ===
plt.figure(figsize=(10, 8))
pos = {node: (last_graph.nodes[node]['x'][0].item(), last_graph.nodes[node]['x'][1].item()) for node in last_graph.nodes()}
scores = [last_graph.nodes[node]['score'] for node in last_graph.nodes()]
normalized_scores = [(s - min(scores)) / (max(scores) - min(scores) + 1e-5) * 0.7 + 0.3 for s in scores]
node_colors = [(s, 0, 0) for s in normalized_scores]

nx.draw_networkx_nodes(last_graph, pos, node_size=700, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(last_graph, pos, arrows=True, edge_color='black')
nx.draw_networkx_labels(last_graph, pos, font_size=9)

plt.title("Final Traffic Graph with SNA Metrics & Congestion Scores")
plt.axis('off')
plt.tight_layout()
plt.show()

# === Print Top Vehicles ===
print("\n[INFO] Top 5 Vehicles by Betweenness Centrality:")
sorted_bet = sorted(last_graph.nodes(data=True), key=lambda x: x[1]['betweenness'], reverse=True)
for node, attr in sorted_bet[:5]:
    print(f"{node}: Betweenness = {attr['betweenness']:.4f}, Degree = {attr['degree']:.4f}, Community = {attr['community']}")

# === Export to CSV ===
csv_path = "sna_vehicle_metrics.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Vehicle_ID', 'X', 'Y', 'Speed', 'Score', 'Degree', 'Closeness', 'Betweenness', 'Community'])
    for node, attr in last_graph.nodes(data=True):
        x_val, y_val, speed = attr['x'][0].item(), attr['x'][1].item(), attr['x'][2].item()
        writer.writerow([
            node,
            x_val,
            y_val,
            speed,
            attr['score'],
            attr['degree'],
            attr['closeness'],
            attr['betweenness'],
            attr['community']
        ])

print(f"\n[INFO] SNA metrics and GNN scores exported to file: {csv_path}")

# Cleanup
for v in vehicles:
    if v.is_alive:
        v.destroy()
print("[INFO] Cleanup complete.")
