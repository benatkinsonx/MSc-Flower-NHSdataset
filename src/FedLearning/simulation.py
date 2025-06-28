# simulation.py
import flwr as fl

from server import server_fn
from client import client_fn

from config import NUM_CLIENTS, NUM_ROUNDS

if __name__ == "__main__":
    # Run simulation with your apps
    fl.simulation.start_simulation(
        client_fn=client_fn,
        server_fn=server_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )