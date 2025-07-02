# simulation.py
import flwr as fl
from flwr.simulation import run_simulation
import matplotlib.pyplot as plt
import numpy as np

from config import NUM_CLIENTS, MODEL_TYPE
from server import server_app, test_acc, test_loss
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

# ============================================================================
# PLOT METRICS
# ============================================================================

training_round = np.arange(len(test_acc)) + 1
PLOT_TITLE = f"Federated {MODEL_TYPE} with {NUM_CLIENTS} clients"

plt.figure(1)
plt.plot(training_round, test_acc)
plt.xlabel('Training Round')
plt.ylabel('Test Accuracy')
plt.title(PLOT_TITLE)
plt.grid()

plt.figure(2)
plt.plot(training_round, test_loss)
plt.xlabel('Training Round')
plt.ylabel('Test Loss')
plt.title(PLOT_TITLE)
plt.grid()

plt.show()