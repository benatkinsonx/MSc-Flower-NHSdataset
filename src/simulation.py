import flwr as fl
from flwr.simulation import run_simulation

# Import your server and client apps
from server import server
from client import client

# Configuration
NUM_CLIENTS = 2

if __name__ == "__main__":
    # Backend configuration
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    
    # Create client configurations to ensure proper partition assignment
    client_configs = []
    for i in range(NUM_CLIENTS):
        client_configs.append({"client_id": i})
    
    # Run simulation with explicit client configurations
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )