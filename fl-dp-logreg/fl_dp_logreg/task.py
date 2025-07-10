"""fl-dp-logreg: A Flower / sklearn app."""

import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data globally (cache it)
df = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition data from your custom dataset."""
    global df
    
    if df is None:
        df = pd.read_csv('./data/gbsg.csv', index_col='Unnamed: 0')
        df = df.drop(['pid'], axis=1)
        print(f"DEBUG: Total dataset size: {len(df)}")
    
    print(f"DEBUG: Splitting {len(df)} samples into {num_partitions} partitions")
    
    # Create dataset and partitioner
    ds = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = ds
    
    print(f"DEBUG: Requesting partition {partition_id} out of {num_partitions} total partitions")
    
    if partition_id >= num_partitions:
        print(f"WARNING: partition_id {partition_id} >= num_partitions {num_partitions}, using modulo")
        partition_id = partition_id % num_partitions
    
    # Load the partition
    partition = partitioner.load_partition(partition_id=partition_id)
    partition_df = partition.to_pandas()

    # Separate features and target
    X = partition_df.drop(['status'], axis=1)
    y = partition_df['status']

    # Apply log transformation (your best-performing approach)
    X_log = np.log1p(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_log, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f'Client ID: {partition_id}, Training instances: {len(X_train)} (log-transformed)')
    return X_train, X_test, y_train, y_test

def get_model(penalty: str, local_epochs: int, epsilon: float = 1.0):
    """Create regular sklearn LogisticRegression (stable and reliable)"""
    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )

def add_dp_noise(params, epsilon, sensitivity=0.1):
    """Add Gaussian noise for differential privacy"""
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    
    noise_scale = sensitivity / epsilon
    noisy_params = []
    
    for param in params:
        noise = np.random.normal(0, noise_scale, param.shape)
        noisy_params.append(param + noise)
    
    return noisy_params

def get_model_params(model):
    """Extract model parameters"""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
    return params

def set_model_params(model, params):
    """Set model parameters"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model):
    """Initialize model parameters"""
    global df
    if df is None:
        df = pd.read_csv('./data/gbsg.csv', index_col='Unnamed: 0')
        df = df.drop(['pid'], axis=1)
    
    n_classes = 2
    n_features = len(df.drop(['status'], axis=1).columns)
    
    print(f"Initializing model with {n_features} features and {n_classes} classes (log-transformed)")
    
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((n_classes, n_features))
    
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))