# config.py
NUM_CLIENTS = 3
MIN_NUM_CLIENTS = 3
NUM_ROUNDS = 100
PENALTY = "l2"
# models available: "logistic_regression"
MODEL_TYPE = "logistic_regression"

# epsilon=10 and data_norm=20 give best result
EPSILON = 6
DATA_NORM = 20
C = 0.25
ITER_PER_ROUND = 1