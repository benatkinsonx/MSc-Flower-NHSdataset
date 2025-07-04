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

plt.figure(1, figsize=(10, 6))
plt.plot(training_round, test_acc)
plt.xlabel('Training Round', fontsize=18)
plt.ylabel('Test Accuracy', fontsize=18)
# plt.title(PLOT_TITLE, fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray', labelsize=18)
plt.tick_params(axis='both', which='major', length=6, color='black', labelsize=18)
plt.tick_params(top=True, right=True, direction='in', length=6)
plt.tick_params(which='minor', top=True, right=True, direction='in', length=4)
plt.savefig(f'federated_{MODEL_TYPE}_accuracy_{NUM_CLIENTS}clients.pdf', format='pdf', bbox_inches='tight')

plt.figure(2, figsize=(10, 6))
plt.plot(training_round, test_loss)
plt.xlabel('Training Round', fontsize=18)
plt.ylabel('Test Loss', fontsize=18)
# plt.title(PLOT_TITLE, fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray', labelsize=18)
plt.tick_params(axis='both', which='major', length=6, color='black', labelsize=18)
plt.tick_params(top=True, right=True, direction='in', length=6)
plt.tick_params(which='minor', top=True, right=True, direction='in', length=4)
plt.savefig(f'federated_{MODEL_TYPE}_loss_{NUM_CLIENTS}clients.pdf', format='pdf', bbox_inches='tight')

plt.show()