# server.py

"""sklearnexample: A Flower / scikit-learn app."""

from flwr.common import Context, NDArrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Optional, Dict, Union, Callable
import numpy as np
import flwr as fl
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters, Context
)

from task import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    set_initial_params,
    set_model_params,
)

from config import NUM_CLIENTS, MIN_NUM_CLIENTS, NUM_ROUNDS, PENALTY

# ============================================================================
# FEDERATED LEARNING STRATEGY
# ============================================================================

class FedLearning(Strategy):
    """Custom strategy using built-in FedAvg aggregation"""
    
    def __init__(self, initial_parameters: Optional[Parameters] = None):
        super().__init__()
        self.initial_parameters = initial_parameters
        # Create a FedAvg instance to use its aggregation method
        self.fed_avg = FedAvg()

    def initialize_parameters(self, client_manager: Optional[ClientManager] = None) -> Optional[Parameters]:
        """Initialize global model parameters"""
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager
                      ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for the fit round"""
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=MIN_NUM_CLIENTS)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results using FedAvg's built-in aggregation"""

        if not results:
            print("WARNING: No results received from clients")
            return None, {}
        
        # Use FedAvg's built-in aggregation method!
        aggregated_params, metrics = self.fed_avg.aggregate_fit(server_round, results, failures)
        
        print(f"Aggregated ML model from {len(results)} clients in round {server_round}")
        return aggregated_params, metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the aggregated model"""
        print(f"Round {server_round}: Model aggregated successfully")
        return 0.0, {"round": server_round}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation"""
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=MIN_NUM_CLIENTS)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], 
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using FedAvg's method"""
        return self.fed_avg.aggregate_evaluate(server_round, results, failures)

# ============================================================================
# SERVER APP CONFIGURATION
# ============================================================================

# ============================================================================
# SERVER APP CONFIGURATION
# ============================================================================

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Create initial model and get parameters
    model = create_log_reg_and_instantiate_parameters(PENALTY)
    ndarrays = get_model_parameters(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Use YOUR custom strategy instead of FedAvg
    strategy = FedLearning(initial_parameters=global_model_init)
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)