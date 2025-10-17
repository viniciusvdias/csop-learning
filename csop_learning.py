import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx
import numpy as np

# ---------------------------------------------------
# 1. HELPER FUNCTION: TRIANGLE COUNTER
# ---------------------------------------------------
# This function is updated to work with a subset of nodes
# from a larger graph, as we are not relabeling them.
def count_triangles_in_subgraph(full_edge_index, subset_nodes):
    """
    Counts triangles in a subgraph defined by a subset of nodes.

    Args:
        full_edge_index (torch.Tensor): The edge_index of the entire graph.
        subset_nodes (torch.Tensor): A tensor containing the indices of nodes
                                     in the subgraph.

    Returns:
        int: The number of triangles in the specified subgraph.
    """
    # Create a set for quick lookups
    subset_set = set(subset_nodes.tolist())
    
    # Filter the edge_index to only include edges where both nodes
    # are in the subgraph.
    mask = [u.item() in subset_set and v.item() in subset_set for u, v in full_edge_index.t()]
    subgraph_edge_index = full_edge_index[:, mask]
    
    # To count triangles, we convert to NetworkX, which handles non-contiguous
    # node labels easily.
    G_sub = nx.Graph()
    G_sub.add_nodes_from(subset_nodes.tolist())
    G_sub.add_edges_from(subgraph_edge_index.t().tolist())
    
    # NetworkX's triangles function counts triangles for each node.
    # The total number is the sum of these counts divided by 3.
    num_triangles = sum(nx.triangles(G_sub).values()) // 3
    
    return num_triangles

# ---------------------------------------------------
# 2. SUBGRAPH SAMPLING & DATASET CREATION (TRANSDUCTIVE)
# ---------------------------------------------------
# The dataset now stores the *indices* of the nodes in each subgraph.
class TransductiveSubgraphDataset(InMemoryDataset):
    def __init__(self, root, original_data, num_samples=5000, min_nodes=5, max_nodes=20, transform=None, pre_transform=None):
        self.original_data = original_data
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        num_original_nodes = self.original_data.num_nodes
        full_edge_index = self.original_data.edge_index

        print("Generating subgraph samples...")

        # Build neighbor list once for random walks
        # full_edge_index is expected to be [2, num_edges]
        edge_arr = full_edge_index.detach().cpu().numpy()
        neighbors = {i: [] for i in range(num_original_nodes)}
        for u, v in edge_arr.T:
            neighbors[int(u)].append(int(v))
            neighbors[int(v)].append(int(u))

        while len(data_list) < self.num_samples:
            start_node = int(np.random.randint(0, num_original_nodes))
            num_hops = int(np.random.randint(self.min_nodes, self.max_nodes + 1))

            # Perform a random walk (sequence includes the start node)
            walk = [start_node]
            current = start_node
            for _ in range(num_hops):
                nbrs = neighbors.get(current, [])
                if len(nbrs) == 0:
                    break
                current = int(np.random.choice(nbrs))
                walk.append(current)

            # Create induced subgraph from unique nodes visited in the walk
            subset_nodes = sorted(set(walk))
            subset = torch.tensor(subset_nodes, dtype=torch.long)

            if self.min_nodes <= len(subset) <= self.max_nodes:
                # Calculate the target: number of triangles normalized by number of vertices
                num_triangles = count_triangles_in_subgraph(full_edge_index, subset)
                triangles_per_vertex = float(num_triangles) / float(len(subset))

                # We store a Data object containing the node indices and the normalized target
                subgraph_info = Data(node_indices=subset, y=torch.tensor([triangles_per_vertex], dtype=torch.float))
                data_list.append(subgraph_info)
                print("Remaining samples to generate:", self.num_samples - len(data_list), end='\r')
        
        print(f"Generated {len(data_list)} samples.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ---------------------------------------------------
# 3. GNN MODEL DEFINITION (TRANSDUCTIVE)
# ---------------------------------------------------
# This model first computes embeddings for all nodes, then pools
# the embeddings for the specific nodes in each subgraph.
class TransductiveGNN(torch.nn.Module):
    def __init__(self, num_node_features, embedding_dim=64, hidden_dim=64):
        super().__init__()
        torch.manual_seed(12345)
        
        # Part 1: GNN Encoder to get node embeddings
        self.encoder = GCNConv(num_node_features, embedding_dim)
        
        # Part 2: MLP Predictor to process the pooled subgraph embedding
        self.predictor = Sequential(
            Linear(embedding_dim, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
            Linear(hidden_dim, 1)
        )

    def forward(self, full_x, full_edge_index, subgraph_node_indices, batch_vector):
        # Step 1: Get embeddings for ALL nodes in the entire graph
        # This is done only once per forward pass, regardless of batch size.
        all_node_embeddings = self.encoder(full_x, full_edge_index).relu()

        # Step 2: Select the embeddings for the nodes in the current batch of subgraphs
        subgraph_node_embeddings = all_node_embeddings[subgraph_node_indices]

        # Step 3: Pool the node embeddings to get a single embedding for each subgraph
        # The 'batch_vector' tells the pooling function which nodes belong to which subgraph.
        subgraph_embeddings = global_mean_pool(subgraph_node_embeddings, batch_vector)
        
        # Step 4: Use the MLP to predict the number of triangles from the subgraph embedding
        return self.predictor(subgraph_embeddings)

# ---------------------------------------------------
# 4. ML PIPELINE
# ---------------------------------------------------
def main():
    print("Step 1: Loading the original graph...")
    original_dataset = Planetoid(root='data/', name='Cora')
    original_data = original_dataset[0]
    #original_dataset = KarateClub()
    #original_data = original_dataset[0]
    # Ensure features and labels are float for the model
    original_data.x = original_data.x.float()
    original_data.y = original_data.y.float()
    print("Original Graph Info:", original_data)
    print("-" * 30)
    
    print("Step 2: Generating and loading the subgraph dataset...")
    # Using a different root to avoid conflict with the previous version
    dataset = TransductiveSubgraphDataset(root='data/transductive_dataset', original_data=original_data, num_samples=2000)
    print(f"Dataset created with {len(dataset)} subgraphs.")
    # --- NEW: print distribution of the targets (triangles per vertex) using percentiles ---
    if len(dataset) > 0:
        # collect all target values
        y_vals = np.array([float(d.y.item()) for d in dataset])
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        p_vals = np.percentile(y_vals, percentiles)
        print("Target (triangles per vertex) distribution percentiles:")
        for perc, val in zip(percentiles, p_vals):
            print(f"  {perc:3d}th percentile: {val:.4f}")
        print(f"  Count: {len(y_vals)}, Min: {y_vals.min():.4f}, Max: {y_vals.max():.4f}, Mean: {y_vals.mean():.4f}, Std: {y_vals.std():.4f}")
    else:
        print("Dataset is empty, cannot compute target distribution.")
    print("-" * 30)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # The DataLoader will automatically batch the 'node_indices' and create a 'batch' vector
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    print("Step 3: Setting up the model and training components...")
    model = TransductiveGNN(num_node_features=dataset.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    print("Model:", model)
    print("-" * 30)

    # Training loop
    print("Step 4: Starting model training...")
    for epoch in range(1, 201): # Increased epochs for better embedding learning
        model.train()
        total_loss = 0
        for subgraph_batch in train_loader:
            optimizer.zero_grad()
            
            # The model needs the full graph AND the batch of subgraph indices
            out = model(
                full_x=original_data.x, 
                full_edge_index=original_data.edge_index, 
                subgraph_node_indices=subgraph_batch.node_indices, 
                batch_vector=subgraph_batch.batch
            )
            
            loss = criterion(out, subgraph_batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * subgraph_batch.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        print(f'Epoch: {epoch:03d}, Avg. Train Loss: {avg_train_loss:.4f}')

    print("Training finished.")
    print("-" * 30)

    # Evaluation loop
    print("Step 5: Evaluating the model on the test set...")
    model.eval()
    total_error = 0
    with torch.no_grad():
        for subgraph_batch in test_loader:
            out = model(
                full_x=original_data.x, 
                full_edge_index=original_data.edge_index, 
                subgraph_node_indices=subgraph_batch.node_indices, 
                batch_vector=subgraph_batch.batch
            )
            error = (out - subgraph_batch.y.view(-1, 1)).abs().sum().item()
            total_error += error

    avg_test_error = total_error / len(test_loader.dataset)
    print(f'Average Test Error (MAE, triangles per vertex): {avg_test_error:.4f}')
    print("-" * 30)
    
    # ---------------------------------------------------
    # 5. EVALUATION loop
    # ---------------------------------------------------
    # No changes here, just keeping the section for clarity
    print("Step 6: Running inference examples (10 randomly generated random-walk subgraphs)...")

    # Build neighbor list (same as in process) to perform random walks
    edge_arr = original_data.edge_index.detach().cpu().numpy()
    num_original_nodes = original_data.num_nodes
    neighbors = {i: [] for i in range(num_original_nodes)}
    for u, v in edge_arr.T:
        neighbors[int(u)].append(int(v))
        neighbors[int(v)].append(int(u))

    # Generate 10 valid random-walk induced subgraphs for inference
    inference_examples = []
    attempts = 0
    while len(inference_examples) < 10 and attempts < 1000:
        attempts += 1
        start_node = int(np.random.randint(0, num_original_nodes))
        num_hops = int(np.random.randint(dataset.min_nodes, dataset.max_nodes + 1))

        walk = [start_node]
        current = start_node
        for _ in range(num_hops):
            nbrs = neighbors.get(current, [])
            if len(nbrs) == 0:
                break
            current = int(np.random.choice(nbrs))
            walk.append(current)

        subset_nodes = sorted(set(walk))
        if dataset.min_nodes <= len(subset_nodes) <= dataset.max_nodes:
            subset = torch.tensor(subset_nodes, dtype=torch.long)
            num_triangles = count_triangles_in_subgraph(original_data.edge_index, subset)
            triangles_per_vertex = float(num_triangles) / float(len(subset))
            subgraph_info = Data(node_indices=subset, y=torch.tensor([triangles_per_vertex], dtype=torch.float))
            inference_examples.append(subgraph_info)

    if len(inference_examples) == 0:
        print("No valid inference examples generated. Check min_nodes/max_nodes or graph connectivity.")
    else:
        # Batch all inference examples together (single forward pass)
        inference_loader = DataLoader(inference_examples, batch_size=len(inference_examples))
        model.eval()
        with torch.no_grad():
            for batch in inference_loader:
                preds = model(
                    full_x=original_data.x,
                    full_edge_index=original_data.edge_index,
                    subgraph_node_indices=batch.node_indices,
                    batch_vector=batch.batch
                ).view(-1).tolist()

                # Split concatenated node_indices using batch vector to print per-example node lists
                results = []
                abs_errors = []
                for i in range(batch.num_graphs):
                    mask = (batch.batch == i)
                    node_idxs = batch.node_indices[mask].tolist()
                    actual_ratio = float(batch.y[i].item())
                    predicted_ratio = float(preds[i])
                    results.append((node_idxs, actual_ratio, predicted_ratio))
                    abs_errors.append(abs(predicted_ratio - actual_ratio))

                # Print all examples (showing ratio and approximate triangle count)
                for i, (nodes, actual, predicted) in enumerate(results, 1):
                    approx_actual_tri = round(actual * len(nodes))
                    approx_pred_tri = round(predicted * len(nodes))
                    print(f"Example {i:02d}:")
                    print(f"  -> Node indices: {nodes}")
                    print(f"  -> Actual triangles/vertex:   {actual:.4f}  (≈ {approx_actual_tri} triangles total)")
                    print(f"  -> Predicted triangles/vertex:{predicted:.4f}  (≈ {approx_pred_tri} triangles total)")
                    print("-" * 20)

                # Compute and print MAE for the inference examples
                if len(abs_errors) > 0:
                    mae = sum(abs_errors) / len(abs_errors)
                    print(f"Inference MAE over {len(abs_errors)} examples: {mae:.4f} (triangles per vertex)")
                else:
                    print("No inference errors to compute MAE.")


if __name__ == '__main__':
    main()
