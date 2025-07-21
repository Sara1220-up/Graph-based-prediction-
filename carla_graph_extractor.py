import carla
import time
import networkx as nx
import math
import torch
from torch_geometric.utils import from_networkx

# ==== CONNECT TO CARLA ====
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# ==== CLEANUP EXISTING VEHICLES ====
existing_vehicles = world.get_actors().filter('vehicle.*')
for vehicle in existing_vehicles:
    vehicle.destroy()
print("♻️ Cleared existing vehicles from the world.")

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

# Add edges based on proximity
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

    if neighbor_count > 0:
        avg_edge_distance = sum(graph[node][nbr]['weight'] for nbr in neighbors) / neighbor_count
    else:
        avg_edge_distance = 0.0

    x = graph.nodes[node]['x']
    y = graph.nodes[node]['y']
    speed = graph.nodes[node]['speed']

    # New 5-feature vector: [x, y, speed, neighbor_count, avg_edge_distance]
    graph.nodes[node]['x'] = torch.tensor([
        x, y, speed, neighbor_count, avg_edge_distance
    ], dtype=torch.float)

# ==== VISUALIZE GRAPH IN CARLA ====
for vehicle in vehicles:
    loc = vehicle.get_transform().location
    world.debug.draw_string(
        location=loc,
        text=f"ID:{vehicle.id}",
        life_time=30.0,
        color=carla.Color(255, 0, 0),
        persistent_lines=True
    )

for edge in graph.edges():
    n1, n2 = edge
    v1 = next((v for v in vehicles if f"vehicle_{v.id}" == n1), None)
    v2 = next((v for v in vehicles if f"vehicle_{v.id}" == n2), None)
    if v1 and v2:
        loc1 = v1.get_transform().location
        loc2 = v2.get_transform().location
        world.debug.draw_line(
            begin=loc1,
            end=loc2,
            thickness=0.5,
            color=carla.Color(0, 255, 0),
            life_time=30.0,
            persistent_lines=True
        )

# ==== CONVERT TO PYTORCH GEOMETRIC DATA ====
pyg_data = from_networkx(graph)
pyg_data.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes()])

print("\n✅ PyG Graph Created!")
print("Node features shape:", pyg_data.x.shape)
print("Edge index shape:", pyg_data.edge_index.shape)

# ==== WAIT TO VIEW VISUALIZATION ====
print("\n⌛ Waiting 30 seconds to view visualization...")
time.sleep(30)

# ==== CLEAN UP ====
for v in vehicles:
    v.destroy()

print("\n✅ Vehicles destroyed. Visualization complete.")
