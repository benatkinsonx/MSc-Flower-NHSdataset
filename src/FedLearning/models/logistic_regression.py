# models/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from flwr.common import NDArrays
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import NUM_FEATURES

def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

# ============================================================================
# MODEL CREATION FUNCTIONS
# ============================================================================

def set_initial_params(model: LogisticRegression) -> None:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([0,1])

    model.coef_ = np.zeros((1, NUM_FEATURES))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))


def create_log_reg_and_instantiate_parameters(penalty):
    """Helper function to create a LogisticRegression model."""
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # client trains for one epoch, sends model updates
        warm_start=True,  # prevent refreshing weights when fitting,
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    return model