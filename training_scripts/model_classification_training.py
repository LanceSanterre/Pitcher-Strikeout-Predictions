"""
model_classifcation_training.py

This script trains multiple classification models (neural networks and tree-based classifiers)
on pitcher data for binary classification. Results and artifacts are logged using MLflow.

Models included:
- PyTorch-based DeepModel_class and TransformerModel
- XGBoostClassifier
- RandomForestClassifier

Author: Lance Santerre
"""

import os
import joblib
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.pytorch
from xgboost import XGBClassifier

from models import DeepModel_class, TransformerModel


# === Custom Dataset for Classification (No embeddings) ===
class StrikeoutDataset(Dataset):
    """
    Custom Dataset for feeding continuous features into a PyTorch classification model.
    """

    def __init__(self, df, cont_cols, target_col):
        self.cont_data = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32).view(
            -1, 1
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.cont_data[idx], self.target[idx]


# === Training Function (No categorical input) ===
def run_single_classification_model(
    X_cleaned,
    y_class,
    model_class,
    model_architecture="deep",
    optimizer_name="adamw",
    learning_rate=0.001,
    scaler_name="robust",
    epochs=100,
    experiment_name="classification_ensemble",
    model_save_path="models/cls",
):
    """
    Trains a single neural network classification model and logs performance to MLflow.

    Returns:
        dict: Model metadata and performance metrics.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cont_cols = X_cleaned.columns.tolist()
    batch_size = 64

    scaler_map = {
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }
    optimizer_map = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}

    scaler = scaler_map[scaler_name]
    optimizer_cls = optimizer_map[optimizer_name]

    os.makedirs(model_save_path, exist_ok=True)
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_architecture}_classification_{tag}"):

        # Create training + validation splits
        df_nn = X_cleaned.copy()
        df_nn["target"] = y_class["target"]

        # Apply scaling
        train_df, val_df = train_test_split(df_nn, test_size=0.2, random_state=42)
        train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols])
        val_df[cont_cols] = scaler.transform(val_df[cont_cols])

        # Initialize model and training objects
        train_ds = StrikeoutDataset(train_df, cont_cols, "target")
        val_ds = StrikeoutDataset(val_df, cont_cols, "target")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = model_class(n_cont=len(cont_cols), output_type="classification").to(
            device
        )

        criterion = nn.BCELoss()
        optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

        # === Training Loop ===
        for epoch in range(epochs):
            model.train()
            for cont_x, y in train_loader:
                cont_x, y = cont_x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(cont_x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # === Evaluation ===
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for cont_x, y in val_loader:
                cont_x = cont_x.to(device)
                preds = model(cont_x).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        bin_preds = (np.array(all_preds) > 0.5).astype(int)
        acc = accuracy_score(all_targets, bin_preds)
        f1 = f1_score(all_targets, bin_preds)
        precision = precision_score(all_targets, bin_preds)
        recall = recall_score(all_targets, bin_preds)

        mlflow.log_metrics(
            {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        )

        # Save model and artifacts
        base_filename = f"{model_architecture}_{optimizer_name}_{scaler_name}_{tag}"
        torch.save(
            model.state_dict(),
            os.path.join(model_save_path, f"{base_filename}_model.pt"),
        )
        joblib.dump(
            scaler, os.path.join(model_save_path, f"{base_filename}_scaler.pkl")
        )
        joblib.dump(
            {"cont_cols": cont_cols},
            os.path.join(model_save_path, f"{base_filename}_features.pkl"),
        )
        joblib.dump(
            {
                "num_cont_features": len(cont_cols),
            },
            os.path.join(model_save_path, f"{base_filename}_metadata.pkl"),
        )

        return {
            "model": model_architecture,
            "optimizer": optimizer_name,
            "scaler": scaler_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "artifact_tag": tag,
        }


# === Tree-Based Model Training (XGBoost or Random Forest) ===
def run_tree_model(
    X_cleaned, y, model_type="xgboost", scaler_name="none", tag_suffix=""
):
    """
    Trains a tree-based model (XGBoost or Random Forest) and logs performance to MLflow.

    Returns:
        dict: Model metadata and performance metrics.
    """
    scaler_map = {
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "none": None,
    }

    os.makedirs("models/cls", exist_ok=True)
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + tag_suffix

    mlflow.set_experiment("classification_ensemble")
    with mlflow.start_run(run_name=f"{model_type}_{scaler_name}_{tag}"):
        X = X_cleaned.copy()
        y = y.copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = scaler_map[scaler_name]
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        if model_type == "xgboost":
            model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.05,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError("Unsupported model_type")

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        precision = precision_score(y_val, preds)
        recall = recall_score(y_val, preds)

        mlflow.log_metrics(
            {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        )
        model_path = f"/Users/lancesanterre/so_predict/models/class/{model_type}_{scaler_name}_{tag}"
        joblib.dump(model, f"{model_path}_model.pkl")
        if scaler:
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
        joblib.dump(X_cleaned.columns.tolist(), f"{model_path}_features.pkl")

        return {
            "model": model_type.capitalize(),
            "scaler": scaler_name,
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "artifact_tag": tag,
        }


# === Load Data ===
# Load cleaned features and labels
X_cleaned = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/X_train.csv"
)
X_cleaned = X_cleaned.drop(columns="player_id")
if "Unnamed: 0" in X_cleaned.columns:
    X_cleaned = X_cleaned.drop(columns=["Unnamed: 0"])

# Skip the first row manually, which contains the string header
y_class = pd.read_csv(
    "/Users/lancesanterre/so_predict/data/training/new_training/y_5_train.csv",
    skiprows=1,
    header=None,
)

# Keep only the second column (label column)
y_class = y_class[[1]]

# Rename the column for downstream compatibility
y_class.columns = ["target"]

# Ensure it's numeric (safety)
y_class["target"] = pd.to_numeric(y_class["target"], errors="raise")

model_save_path = "/Users/lancesanterre/so_predict/models/class"

# === Run XGBoost model ===
print("🔁 Training Random Forest (tuned)")
result_rf_tuned = run_tree_model(
    X_cleaned,
    y_class["target"],
    model_type="random_forest",
    scaler_name="standard",
    tag_suffix="_rf_tuned",
)
print("✅", result_rf_tuned)
print("-" * 80)

print("🔁 Training XGBoost (better config)")
result_xgb_better = run_tree_model(
    X_cleaned,
    y_class["target"],
    model_type="xgboost",
    scaler_name="none",  # 🚫 Don't scale for XGBoost
    tag_suffix="_xgb_better",
)
print("✅", result_xgb_better)
print("-" * 80)

print("🔁 Training XGBoost (better config)")
result_xgb_better = run_tree_model(
    X_cleaned,
    y_class["target"],
    model_type="xgboost",
    scaler_name="standard",  # 🚫 Don't scale for XGBoost
    tag_suffix="_xgb_better_2",
)
print("✅", result_xgb_better)
print("-" * 80)


print("🔁 Training RF (better config)")
result_xgb_better = run_tree_model(
    X_cleaned,
    y_class["target"],
    model_type="random_forest",
    scaler_name="robust",  # 🚫 Don't scale for XGBoost
    tag_suffix="_rf_better_2_robust",
)
print("✅", result_xgb_better)
print("-" * 80)

print("🔁 Training XGBoost (better config)")
result_xgb_better = run_tree_model(
    X_cleaned,
    y_class["target"],
    model_type="xgboost",
    scaler_name="minmax",  # 🚫 Don't scale for XGBoost
    tag_suffix="_xgb_better_2_min",
)
print("✅", result_xgb_better)
print("-" * 80)
