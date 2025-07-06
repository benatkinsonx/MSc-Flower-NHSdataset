
# Updated client.py
"""sklearnexample: A Flower / scikit-learn app with DiffPrivLib."""

import warnings
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from sklearn.metrics import log_loss

from models.logistic_regression import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    set_model_params
)

from config import MODEL_TYPE, EPSILON, DATA_NORM
from dataloader import df, load_datasets

# ============================================================================
# FLOWER CLIENT IMPLEMENTATION WITH DIFFPRIVLIB
# ============================================================================

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        
        # DiffPrivLib handles DP automatically during training!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # No need for manual DP - it's built into the optimization
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

# ============================================================================
# CLIENT APP CONFIGURATION
# ============================================================================

def client_app(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_datasets(df, num_partitions, partition_id)

    # Get DP configuration from run_config
    epsilon = EPSILON
    data_norm = DATA_NORM
    
    # # Estimate data_norm if not provided (causes privacy leakage!)
    # if data_norm is None:
    #     import numpy as np
    #     data_norm = np.linalg.norm(X_train, axis=1).max()
    #     print(f"Client {partition_id}: Estimated data_norm = {data_norm:.3f}")

    if MODEL_TYPE == 'logistic_regression':
        model = create_log_reg_and_instantiate_parameters(
            epsilon=epsilon,
            data_norm=data_norm
        )
        print(f"Client {partition_id}: Created DP LogReg with ε={epsilon}, data_norm={data_norm:.3f}")
        
    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()
    
# Create ClientApp
client_app = ClientApp(client_fn=client_app)