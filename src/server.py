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

class FedAnalytics(Strategy):
    def __init__(self):
        super().__init__()
        self.num_clients = 2

    def initialize_parameters(self, client_manager: Optional[ClientManager] = None) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=2)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters))[0][0] for _, fit_res in results
        ]
        mean_agg = sum(values_aggregated)/self.num_clients
        results_array = np.array([mean_agg])
        return ndarrays_to_parameters(results_array), {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        agg_mean = [arr.item() for arr in parameters_to_ndarrays(parameters)]
        return 0, {"Aggregated mean age": agg_mean}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        pass

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    strategy = FedAnalytics()
    config = ServerConfig(num_rounds=1)
    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp (modern approach)
server = ServerApp(server_fn=server_fn)

# Legacy support for direct execution
if __name__ == "__main__":
    # Use the legacy start_server for direct execution
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=1),
        strategy=FedAnalytics(),
    )