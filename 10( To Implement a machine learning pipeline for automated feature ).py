import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# -------------------------------
# LOAD DATA
# -------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -------------------------------
# TRAIN/TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# DEFINE MODELS AND PARAM GRID
# -------------------------------
pipelines = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),

    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier())
    ]),

    "gradient_boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier())
    ])
}

param_grids = {
    "logistic": {
        "clf__C": [0.01, 0.1, 1, 10]
    },

    "random_forest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None]
    },

    "gradient_boosting": {
        "clf__n_estimators": [50, 100],
        "clf__learning_rate": [0.01, 0.1, 0.2]
    }
}

# -------------------------------
# AUTOMATED MODEL SELECTION
# -------------------------------
best_model = None
best_score = 0
results = []

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")

    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

    results.append({
        "model": name,
        "best_params": grid.best_params_,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    })

    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
results_df = pd.DataFrame(results)
print("\nAll Model Results:")
print(results_df)

print("\nBest Model:")
print(best_model)

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBest model saved as best_model.pkl")
