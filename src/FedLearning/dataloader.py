# dataloader.py

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset

# ============================================================================
# DATA LOADING & PARTITIONING
# ============================================================================

# Load data globally
df = pd.read_csv('./data/gbsg.csv', index_col='Unnamed: 0')
df = df.drop(['pid'], axis=1)

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

    X = partition_df.drop(['status'], axis=1)
    y = partition_df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'client ID: {client_id}, no. of training instances: {len(X_train)}')
    return X_train, X_test, y_train, y_test

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

NUM_CLASSES = 2
NUM_FEATURES = len(df.drop(['status'], axis=1).columns)