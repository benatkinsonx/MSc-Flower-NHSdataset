# client.py - Simple tutorial version

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
import flwr as fl
import os

print(f"DEBUG: CLIENT_ID environment variable = '{os.getenv('CLIENT_ID')}'")
client_id = int(os.getenv('CLIENT_ID', '0'))
print(f"DEBUG: Parsed client_id = {client_id}")

df = pd.read_csv("./data/gbsg.csv")

def load_datasets(df, num_partitions: int, client_id: int):
    ds = Dataset.from_pandas(df) # convert pd df to hugging face dataset

    # create partitioner
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = ds

    partition = partitioner.load_partition(partition_id=client_id) # load specific partition
    partition_df = partition.to_pandas() # convert the partition back to pd df

    print(f'client ID: {client_id}, no. of instances: {len(partition_df)}')

    return partition_df


def compute_mean(df, col_name):
    """Compute histogram for a given column"""
    return df[col_name].mean()

class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated analytics"""

    # In client.py
    def fit(self, parameters, config):
        partition_df = load_datasets(df, num_partitions=2, client_id=client_id)
        print(f"Client dataset size: {len(partition_df)}")
        print(f"First 5 patient IDs: {partition_df.index[:5].tolist()}")
        print(f"Mean age: {partition_df['age'].mean()}")
        
        partition_mean = compute_mean(partition_df, 'age')
        return ([np.array([partition_mean])], len(partition_df), {})

# Simple tutorial approach
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )