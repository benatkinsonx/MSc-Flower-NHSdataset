from typing import List, Tuple, Optional, Dict, Union, Callable
import numpy as np

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

class FedAnalytics(Strategy):
    def __init__(
        self, compute_fns: List[Callable] = None, col_names: List[str] = None
    ) -> None:
        
        super().__init__()

    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
    
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        return [(client, fit_ins) for client in clients]
    

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Get results from fit
        # Convert results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]
        length_agg_hist = 0
        width_agg_hist = 0
        for val in values_aggregated:
            length_agg_hist += val[0]
            width_agg_hist += val[1]

        ndarr = np.concatenate((["Length:"], length_agg_hist, ["Width:"], width_agg_hist))
        return ndarrays_to_parameters(ndarr), {}
    

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        agg_hist = [arr.item() for arr in parameters_to_ndarrays(parameters)]
        return 0, {"Aggregated histograms": agg_hist}
    

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    
        pass


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    
        pass


if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=1),
        strategy=FedAnalytics(),
    )