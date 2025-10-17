# CSOP Learning: Graph Neural Networks for Subgraph Triangle Counting

A machine learning project that uses Graph Neural Networks (GNNs) to predict the number of triangles in graph subgraphs. This implementation demonstrates a transductive learning approach using PyTorch Geometric.

## Overview

This project implements a GNN-based solution for counting triangles in subgraphs of larger graphs. The approach uses:
- **Transductive learning**: The model learns embeddings for all nodes in the entire graph
- **Subgraph sampling**: Random walk-based sampling to create training subgraphs
- **Triangle counting**: Predicts triangles per vertex as a normalized metric

### What is Triangle Counting?

Triangle counting is a fundamental graph analysis task that identifies closed triads (three interconnected nodes) in a graph. This metric is useful for:
- Community detection
- Social network analysis
- Graph clustering coefficient computation
- Detecting dense subgraphs

## Features

- **Transductive GNN Architecture**: Computes node embeddings for the entire graph, then pools specific subgraph embeddings
- **Random Walk Sampling**: Generates diverse training subgraphs using random walk exploration
- **Flexible Subgraph Sizes**: Configurable minimum and maximum subgraph sizes
- **Comprehensive Evaluation**: Includes training, testing, and inference examples with detailed metrics
- **Target Distribution Analysis**: Provides percentile statistics for dataset understanding

## Requirements

### Python Dependencies

```
torch
torch-geometric
networkx
numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/viniciusvdias/csop-learning.git
cd csop-learning
```

2. Install dependencies:
```bash
pip install torch torch-geometric networkx numpy
```

3. Run the main script:
```bash
python csop_learning.py
```

## Usage

### Basic Execution

Simply run the main Python script:

```bash
python csop_learning.py
```

This will:
1. Load the Cora citation network dataset
2. Generate 2,000 subgraph samples using random walks
3. Train the GNN model for 200 epochs
4. Evaluate on test set
5. Run inference on 10 new randomly generated subgraphs

### Configuration Options

You can modify the following parameters in the `main()` function:

- **Dataset**: Change from Cora to KarateClub or other Planetoid datasets
- **Number of samples**: Adjust `num_samples` in `TransductiveSubgraphDataset`
- **Subgraph size**: Modify `min_nodes` and `max_nodes` parameters
- **Training epochs**: Change the range in the training loop
- **Batch size**: Adjust in `DataLoader` initialization
- **Learning rate**: Modify in `optimizer` initialization

## Project Structure

```
csop-learning/
├── README.md                    # This file
├── csop_learning.py            # Main implementation
└── data/                       # Generated dataset directory (created at runtime)
    ├── Cora/                   # Original graph dataset
    └── transductive_dataset/   # Generated subgraph samples
```

## Technical Details

### Model Architecture

The `TransductiveGNN` model consists of two main components:

1. **GNN Encoder** (`GCNConv`):
   - Processes the entire graph to produce node embeddings
   - Single GCN layer with ReLU activation
   - Embedding dimension: 64 (default)

2. **MLP Predictor**:
   - Takes pooled subgraph embeddings
   - Architecture: Linear → ReLU → BatchNorm → Linear
   - Output: Single value (triangles per vertex)

### Dataset Generation

The `TransductiveSubgraphDataset` class:
- Uses random walks to sample connected subgraphs
- Computes ground truth triangle counts using NetworkX
- Normalizes by vertex count to handle variable-sized subgraphs
- Stores node indices rather than full subgraph structures

### Training Process

1. **Forward Pass**: 
   - Compute embeddings for all nodes in the original graph
   - Extract embeddings for nodes in each subgraph batch
   - Pool subgraph node embeddings (mean pooling)
   - Predict triangles per vertex

2. **Loss Function**: Mean Squared Error (MSE)

3. **Optimizer**: Adam with learning rate 0.01

### Evaluation Metrics

- **Training Loss**: Average MSE over training set
- **Test Error**: Mean Absolute Error (MAE) in triangles per vertex
- **Inference Examples**: Shows predictions vs. ground truth for 10 random subgraphs

## Example Output

```
Step 1: Loading the original graph...
Original Graph Info: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], ...)

Step 2: Generating and loading the subgraph dataset...
Generated 2000 samples.
Target (triangles per vertex) distribution percentiles:
  ...

Step 3: Setting up the model and training components...
Model: TransductiveGNN(...)

Step 4: Starting model training...
Epoch: 001, Avg. Train Loss: 2.3456
...

Step 5: Evaluating the model on the test set...
Average Test Error (MAE, triangles per vertex): 0.1234

Step 6: Running inference examples...
Example 01:
  -> Node indices: [45, 123, 456, ...]
  -> Actual triangles/vertex:   1.2000  (≈ 12 triangles total)
  -> Predicted triangles/vertex: 1.1500  (≈ 11 triangles total)
```

## Customization

### Using Different Graphs

To use KarateClub dataset instead of Cora:

```python
# Comment out Cora loading
# original_dataset = Planetoid(root='data/', name='Cora')
# original_data = original_dataset[0]

# Uncomment KarateClub loading
original_dataset = KarateClub()
original_data = original_dataset[0]
```

### Adjusting Model Capacity

Modify the model initialization:

```python
model = TransductiveGNN(
    num_node_features=dataset.num_node_features,
    embedding_dim=128,  # Increase embedding size
    hidden_dim=128      # Increase hidden layer size
)
```

## Future Improvements

- [ ] Add support for multiple GCN layers
- [ ] Implement attention mechanisms for pooling
- [ ] Add cross-validation for hyperparameter tuning
- [ ] Support for directed graphs
- [ ] GPU acceleration option
- [ ] Visualization of learned embeddings
- [ ] Export trained models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Acknowledgments

- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Uses the Cora citation network dataset from the Planetoid collection
- Triangle counting implementation using NetworkX
