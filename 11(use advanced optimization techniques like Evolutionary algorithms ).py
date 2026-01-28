# -----------------------------------------
# FINAL BAYESIAN OPTIMIZATION SCRIPT
# -----------------------------------------

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    raise ImportError("Install first: pip install bayesian-optimization")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_dim = X.shape[1]

# ---------------------------------------------------
# REAL evaluate_network FUNCTION
# ---------------------------------------------------
def evaluate_network(dropout, learning_rate, neuronPct, neuronShrink):
    """
    Builds, trains and evaluates a neural network.
    Returns NEGATIVE validation loss for Bayesian Optimization.
    """

    # Convert continuous hyperparams to valid sizes
    base_neurons = int(neuronPct * 100)
    second_neurons = int(base_neurons * neuronShrink)

    # Safety bounds
    base_neurons = max(4, base_neurons)
    second_neurons = max(2, second_neurons)

    # Build model
    model = Sequential([
        Dense(base_neurons, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(second_neurons, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train quietly
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=0,
        validation_data=(X_val, y_val)
    )

    # Get final validation loss
    val_loss = history.history["val_loss"][-1]

    # BayesianOptimization maximizes -> return NEGATIVE loss
    return -val_loss


# ---------------------------------------------------
# PARAMETER BOUNDS (from your PDF)
# ---------------------------------------------------
pbounds = {
    "dropout": (0.0, 0.499),
    "learning_rate": (1e-6, 0.1),
    "neuronPct": (0.01, 1),
    "neuronShrink": (0.01, 1)
}

# ---------------------------------------------------
# Create optimizer
# ---------------------------------------------------
optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)

# ---------------------------------------------------
# Run optimization
# ---------------------------------------------------
print("\n🚀 Starting Bayesian Optimization...")
start_time = time.time()

optimizer.maximize(
    init_points=5,
    n_iter=10
)

elapsed = time.time() - start_time

# ---------------------------------------------------
# Print results
# ---------------------------------------------------
print("\n⏱️ Total Runtime: {:.2f} seconds".format(elapsed))
print("🔥 BEST RESULT FOUND:")
print(optimizer.max)

best_params = optimizer.max["params"]
best_score = optimizer.max["target"]

print("\nBest Score:", best_score)
print("Best Parameters:", best_params)
