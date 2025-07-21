import carla
import time
import networkx as nx
import math
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Setup CARLA connection
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Create folder to save snapshots
os.makedirs("graph_dataset", exist_ok=True)

# Number of snapshots to generate
NUM_SNAPSHOTS = 30

for snap in range(NUM_SNAPSHOTS):
    print(f"\n📦 Generating snapshot {snap + 1}/{NUM_SNAPSHOTS}...")

    # Clear existing vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    # Spawn vehicles
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicles = []
    for i in range(5):
        if i >= len(spawn_points):
            break
        v = world.try_spawn_actor(vehicle_bp, spawn_points[i])
        if v:
            v.set_autopilot(True)
            vehicles.append(v)

    if not vehicles:
        print("❌ No vehicles spawned. Skipping snapshot.")
        continue

    time.sleep(6)  # Let vehicles move

    # Build graph
    graph = nx.DiGraph()
    for v in vehicles:
        tf = v.get_transform()
        vel = v.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        graph.add_node(f"vehicle_{v.id}", x=tf.location.x, y=tf.location.y, speed=speed)

    # Add edges based on distance
    nodes = list(graph.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, d1 = nodes[i]
            n2, d2 = nodes[j]
            dist = math.hypot(d1['x'] - d2['x'], d1['y'] - d2['y'])
            if dist < 50:
                graph.add_edge(n1, n2, weight=dist)

    # Convert to PyG
    for node in graph.nodes(data=True):
        graph.nodes[node[0]]['x'] = torch.tensor([
            node[1]['x'], node[1]['y'], node[1]['speed']
        ], dtype=torch.float)

    data = from_networkx(graph)
    data.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes()])

    # Create dummy labels: 1 if speed < 2.0
    labels = [1 if graph.nodes[n]['speed'] < 2.0 else 0 for n in graph.nodes()]
    data.y = torch.tensor(labels, dtype=torch.float).view(-1, 1)

    # Save the graph
    torch.save(data, f"graph_dataset/graph_{snap + 1}.pt")
    print(f"✅ Saved: graph_dataset/graph_{snap + 1}.pt")

    # Cleanup
    for v in vehicles:
        v.destroy()

print("\n✅ Dataset generation complete!")
