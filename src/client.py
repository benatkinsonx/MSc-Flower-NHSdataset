from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# ============== PARTITIONING DATA + DEFINING SUMMARY STATISTIC ==============

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

def compute_mean(df, col_name):
    """Compute mean for a given column"""
    return df[col_name].mean()

# ============== FLOWER CLIENT ==============

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

        summarystat = [np.array(partition_mean)]
        num_examples = len(partition_df)
        metrics = {}
        return (summarystat, num_examples, metrics)
    
# ============== CREATE CLIENT APP ==============

def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    
    # Use hash of node_id to get consistent but different client IDs
    import hashlib
    node_str = str(context.node_id)
    hash_val = int(hashlib.md5(node_str.encode()).hexdigest(), 16)
    client_id = hash_val % 2  # For 2 partitions
    
    print(f"DEBUG: Node {context.node_id} -> Client ID {client_id}")
    
    # Create and return the client
    return FlowerClient(client_id=client_id).to_client()

# Create the ClientApp (modern approach)
client = ClientApp(client_fn=client_fn)

# ============== FOR RUNNING client.py ==============

# Legacy support for direct execution
if __name__ == "__main__":
    import os
    client_id = int(os.getenv('CLIENT_ID', '0'))
    
    # Use the legacy start_client for direct execution
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id=client_id).to_client(),
    )