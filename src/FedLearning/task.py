# task.py

"""sklearnexample: A Flower / scikit-learn app."""

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
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

# ============================================================================
# PARAMETER MANAGEMENT FUNCTIONS
# ============================================================================

def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

# ============================================================================
# MODEL CREATION FUNCTIONS
# ============================================================================

def set_initial_params(model: LogisticRegression) -> None:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([0,1])

    model.coef_ = np.zeros((1, NUM_FEATURES))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))


def create_log_reg_and_instantiate_parameters(penalty):
    """Helper function to create a LogisticRegression model."""
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # client trains for one epoch, sends model updates
        warm_start=True,  # prevent refreshing weights when fitting,
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    return model