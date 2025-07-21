# Graph-based-prediction-
# Rewriting README.md to include content based on the user's LaTeX report
 🚗 Graph-Based Real-Time Traffic Flow Prediction using GNN and SNA

This project presents a novel hybrid approach to predict real-time traffic congestion by modeling vehicle interactions as dynamic graphs using Graph Neural Networks (GNNs) and Social Network Analysis (SNA). It uses simulation data from the CARLA environment and processes graph-based representations of vehicles to perform accurate and interpretable congestion predictions.

---

## 🧠 Core Idea

Autonomous vehicle interactions are modeled as evolving graphs:
- Nodes = Vehicles
- Edges = Spatial proximity (<50 meters)
- Features = Position, speed, neighbor count, avg distance

The pipeline applies:
- **GCN (Graph Convolutional Network)** for congestion prediction
- **Social Network Analysis (SNA)** for interpretability

---

## 🛠️ Methodology

### 🔄 Graph Construction
- Data Source: CARLA Simulator (Town10HD)
- Vehicle data collected every 1s
- Nodes carry feature vectors: `[x, y, speed, neighbor_count, avg_distance]`
- Undirected edges drawn based on distance

### 🧠 GCN Architecture
- GCNConv (5 → 16) → ReLU → GCNConv (16 → 1) → Sigmoid
- Trained using Binary Cross Entropy on congestion labels (speed < 2 m/s)

### 📉 SNA Metrics
- **Degree Centrality:** Local density estimation
- **Closeness Centrality:** Communication efficiency
- **Betweenness Centrality:** Bottleneck influence
- **Community Detection:** Label Propagation for subgraph formation

---

## 📊 Results

- Accuracy: **91.2%**
- F1 Score: **0.89**
- Inference time: < 200ms/frame
- Congestion score highly correlated with betweenness centrality (0.74)

---

## 🔍 Key Observations

- Vehicles with high **degree** and low **speed** initiate congestion
- **Betweenness** central nodes are bottlenecks
- Community detection reveals coordinated vehicle groups
- Low-centrality vehicles move freely, ideal for rerouting

---

## 📁 Project Structure

- `src/`: Python files (GCN, SNA, simulation interface)
- `data/`: Vehicle logs and metrics
- `images/`: Visualizations
- `models/`: Saved GNN models

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

python src/data_collector.py
python src/graph_builder.py
python src/gnn_trainer.py
python src/predictor.py
