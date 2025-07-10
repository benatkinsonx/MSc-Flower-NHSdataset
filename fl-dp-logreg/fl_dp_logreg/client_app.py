"""fl-dp-logreg: A Flower / sklearn app."""

import warnings
from sklearn.metrics import log_loss
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_dp_logreg.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
    add_dp_noise,  # Import the DP function
)

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, epsilon=1.0):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.epsilon = epsilon

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Get model parameters
        params = get_model_params(self.model)
        
        # Add differential privacy noise
        if self.epsilon > 0:
            params = add_dp_noise(params, self.epsilon, sensitivity=1.0)
            print(f"Added DP noise with epsilon={self.epsilon}")

        return params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    epsilon = context.run_config.get("epsilon", 1.0)  # Get epsilon from config
    
    model = get_model(penalty, local_epochs, epsilon)

    # Setting initial parameters
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, epsilon).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)