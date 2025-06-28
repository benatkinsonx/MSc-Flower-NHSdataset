# client.py - Simple tutorial version

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Flower imports
import flwr as fl

# Load the data
df = pd.read_csv("./data/gbsg.csv")

def compute_mean(df, col_name):
    """Compute histogram for a given column"""
    return df[col_name].mean()

class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated analytics"""

    # In client.py
    def fit(self, parameters, config):
        print(f"Client dataset size: {len(df)}")
        print(f"First 5 patient IDs: {df.index[:5].tolist()}")
        print(f"Mean age: {df['age'].mean()}")
        
        mean = compute_mean(df, 'age')
        return ([np.array([mean])], len(df), {})

# Simple tutorial approach
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )