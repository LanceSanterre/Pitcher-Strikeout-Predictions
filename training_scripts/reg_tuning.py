"""
reg_tuning.py

Performs hyperparameter optimization for Random Forest and XGBoost regressors using Optuna.
The script loads and scales the dataset, defines objective functions, and evaluates the best models
on a validation set using R², MSE, and RMSE.

Author: Lance Santerre
"""

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# === Load Data ===
X_cleaned = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/X_train.csv"
)
X_cleaned = X_cleaned.drop(columns="player_id")
if "Unnamed: 0" in X_cleaned.columns:
    X_cleaned = X_cleaned.drop(columns=["Unnamed: 0"])
X = X_cleaned
# Skip the first row manually, which contains the string header
y = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/y_train.csv",
    skiprows=1,
    header=None,
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# === Optuna Objective Functions ===
def rf_objective(trial):
    """Defines hyperparameter search space and returns CV score for Random Forest."""
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )
    score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="r2")
    return score.mean()


# === Optuna Objective for XGBoost ===
def xgb_objective(trial):
    """Defines hyperparameter search space and returns CV score for XGBoost."""
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0, 5)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 10.0)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
    )
    score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="r2")
    return score.mean()


# === Run Optimization ===
print("🔍 Tuning Random Forest...")
rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=50)

print("🔍 Tuning XGBoost...")
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)

# === Evaluate Best Models ===
best_rf_params = rf_study.best_params
best_xgb_params = xgb_study.best_params

print("\n=== Best Random Forest Params ===")
print(best_rf_params)

print("\n=== Best XGBoost Params ===")
print(best_xgb_params)

# Train final models on full training set with best params
rf_best = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
rf_best.fit(X_train_scaled, y_train)
rf_preds = rf_best.predict(X_test_scaled)

xgb_best = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
xgb_best.fit(X_train_scaled, y_train)
xgb_preds = xgb_best.predict(X_test_scaled)


# === Metrics ===
def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")


print_metrics("Random Forest", y_test, rf_preds)
print_metrics("XGBoost", y_test, xgb_preds)
