# client.py - Simple tutorial version

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Flower imports
import flwr as fl

# Load the data
df = pd.read_csv("./data/client.csv")

def compute_hist(df, col_name):
    """Compute histogram for a given column"""
    counts, _ = np.histogram(df[col_name])
    return counts

class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated analytics"""

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str] # parameters = the received updated global parameters, config = the received training config --> not needed here because we are doing FA not FL
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for col_name in ["sepal length (cm)", "sepal width (cm)"]:
            hist = compute_hist(df, col_name)
            hist_list.append(hist)
        return (
            hist_list,
            len(df),
            {},
        )

# Simple tutorial approach
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )