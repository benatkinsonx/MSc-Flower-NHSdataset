# config.py
NUM_CLIENTS = 10
MIN_NUM_CLIENTS = 3
NUM_ROUNDS = 100
PENALTY = "l2"
FRACTION_FIT = 0.5
# models available: "logistic_regression"
MODEL_TYPE = "logistic_regression"

# epsilon=10 and data_norm=20 give best result
EPSILON = 10.0
DATA_NORM = 20
C = 1000000
ITER_PER_ROUND = 1