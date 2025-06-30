# simulation.py
import flwr as fl
from flwr.simulation import run_simulation

from config import NUM_CLIENTS

# server and client apps
from server import server_app
from client import client_app


if __name__ == "__main__":
    # Backend configuration
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    
    run_simulation(
        server_app=server_app,
        client_app=client_app, 
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )