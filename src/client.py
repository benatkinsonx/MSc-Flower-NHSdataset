from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import hashlib
import os

# ============================================================================
# DATA LOADING & PARTITIONING
# ============================================================================

# Load data globally
df = pd.read_csv("./data/gbsg.csv")

def load_datasets(df, num_partitions: int, client_id: int):
    ds = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = ds
    
    # Debug information
    print(f"DEBUG: Requesting partition {client_id} out of {num_partitions} total partitions")
    print(f"DEBUG: Available partition IDs should be: 0 to {num_partitions-1}")
    
    # Ensure client_id is within valid range
    if client_id >= num_partitions:
        print(f"WARNING: client_id {client_id} >= num_partitions {num_partitions}, using modulo")
        client_id = client_id % num_partitions
    
    partition = partitioner.load_partition(partition_id=client_id)
    partition_df = partition.to_pandas()
    print(f'client ID: {client_id}, no. of instances: {len(partition_df)}')
    return partition_df

# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================

def compute_mean(df, col_name):
    """Compute mean for a given column"""
    return df[col_name].mean()

# ============================================================================
# FLOWER CLIENT IMPLEMENTATION
# ============================================================================

class FlowerClient(NumPyClient):
    """Flower client for federated analytics"""
    
    def __init__(self, client_id: int):
        self.client_id = client_id

    def fit(self, parameters, config):
        partition_df = load_datasets(df, num_partitions=2, client_id=self.client_id)
        print(f"Client {self.client_id} dataset size: {len(partition_df)}")
        print(f"First 5 patient IDs: {partition_df.index[:5].tolist()}")
        print(f"Mean age: {partition_df['age'].mean()}")
        
        partition_mean = compute_mean(partition_df, 'age')

        summarystat = [np.array([partition_mean])]
        num_examples = len(partition_df)
        metrics = {}
        return (summarystat, num_examples, metrics)
    
# ============================================================================
# CLIENT APP CONFIGURATION
# ============================================================================

_node_to_client_mapping = {}

def client_fn(context: Context) -> fl.client.Client:
    global _node_to_client_mapping
    
    if context.node_id not in _node_to_client_mapping:
        # Assign next available client ID
        NUM_CLIENTS = 2
        _node_to_client_mapping[context.node_id] = len(_node_to_client_mapping) % NUM_CLIENTS
    
    client_id = _node_to_client_mapping[context.node_id]
    print(f"DEBUG: Node {context.node_id} -> Client ID {client_id} (deterministic)")
    
    return FlowerClient(client_id=client_id).to_client()

# Create the ClientApp (modern approach)
client = ClientApp(client_fn=client_fn)

# ============================================================================
# FOR RUNNING client.py (legacy)
# ============================================================================

# Legacy support for direct execution
if __name__ == "__main__":
    client_id = int(os.getenv('CLIENT_ID', '0'))
    
    # Use the legacy start_client for direct execution
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id=client_id).to_client(),
    )