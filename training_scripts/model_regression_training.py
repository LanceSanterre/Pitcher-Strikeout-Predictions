"""
model_regression_training.py

This script trains and saves two final regression models (Random Forest and XGBoost)
using previously optimized hyperparameters. It scales the full dataset and persists
the models, scaler, and feature list for future prediction.

Author: Lance Santerre
"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# === Load full dataset ===
X_cleaned = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/X_train.csv"
)
X_cleaned = X_cleaned.drop(columns="player_id")
if "Unnamed: 0" in X_cleaned.columns:
    X_cleaned = X_cleaned.drop(columns=["Unnamed: 0"])
X = X_cleaned

# === Load regression target values ===
# Skips header row which contains string labels
y = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/y_train.csv",
    skiprows=1,
    header=None,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Best found parameters ===
best_rf_params = {
    "n_estimators": 444,
    "max_depth": 22,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
best_xgb_params = {
    "n_estimators": 492,
    "max_depth": 12,
    "learning_rate": 0.013227551359522527,
    "subsample": 0.6910498697083453,
    "colsample_bytree": 0.780494860193651,
    "gamma": 3.1380925621260203,
    "reg_lambda": 6.407545442437894,
    "objective": "reg:squarederror",
}

# === Train final models on full data ===
rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
rf_model.fit(X_scaled, y)

xgb_model = XGBRegressor(**best_xgb_params, random_state=42)
xgb_model.fit(X_scaled, y)

# === Save models, scaler, and feature names ===
model_dir = "/Users/lancesanterre/so_predict/models/reg"

joblib.dump(rf_model, f"{model_dir}/rf_model.pkl")
joblib.dump(xgb_model, f"{model_dir}/xgb_model.pkl")
joblib.dump(scaler, f"{model_dir}/scaler.pkl")
joblib.dump(list(X.columns), f"{model_dir}/features.pkl")

print("✅ Models, scaler, and features saved for future prediction.")
