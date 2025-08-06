"""
regression_pred.py

Loads multiple trained neural network regression models and combines their predictions
using statistical ensemble techniques. Saves a single prediction row to CSV.

Author: Lance Santerre
"""

import torch
import pandas as pd
import joblib
from models import DeepModel_res, ResNetModel, ShallowModel
import numpy as np
from scipy.stats.mstats import winsorize


def reg_class(pitcher):
    """
    Run regression ensemble using 5 different neural network models and save prediction CSV.

    Args:
        pitcher (str): Unique identifier for naming output prediction file.
    """
    # === Model configurations ===
    base_path = "/Users/lancesanterre/so_predict/models/reg"
    input_path = "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv"

    model_map = {
        "deep_1": ("deep_1", DeepModel_res),
        "resnet_1": ("ResNetModel_1", ResNetModel),
        "shallow_1": ("shallow_1", ShallowModel),
        "deep_2": ("deep_2", DeepModel_res),
        "resnet_2": ("ResNetModel_2", ResNetModel),
    }
    # === Load new input ===
    input_df = pd.read_csv(input_path)

    predictions = []

    for name, (file_prefix, ModelClass) in model_map.items():
        # Load artifacts
        scaler = joblib.load(f"{base_path}/{file_prefix}_scaler.pkl")
        feature_info = joblib.load(f"{base_path}/{file_prefix}_features.pkl")
        metadata = joblib.load(f"{base_path}/{file_prefix}_metadata.pkl")

        cont_cols = feature_info["cont_cols"]
        cat_cols = feature_info["cat_cols"]

        # Prepare input
        X_cont = torch.tensor(
            scaler.transform(input_df[cont_cols]), dtype=torch.float32
        )
        X_cat = torch.tensor(input_df[cat_cols].values, dtype=torch.long)

        # Load and run model
        model = ModelClass(
            n_cont=metadata["num_cont_features"],
            n_cat=metadata["embedding_input_dim"],
            emb_dim=metadata["embedding_output_dim"],
            output_type="regression",
        )
        model.load_state_dict(
            torch.load(f"{base_path}/{file_prefix}_regression_model.pt")
        )
        model.eval()

        with torch.no_grad():
            pred = model(X_cont, X_cat)
            print(f"[{name.upper()}] Prediction:", pred.item())
            predictions.append(pred.item())

    predictions = np.array(predictions)

    mean_pred = np.mean(predictions)
    median_pred = np.median(predictions)
    trimmed_mean = np.mean(np.sort(predictions)[1:-1])
    winsor_mean = np.mean(winsorize(predictions, limits=[0.2, 0.2]))

    weights = [0.25, 0.2, 0.1, 0.25, 0.2]  # match 5 models
    weighted_avg = np.average(predictions, weights=weights)

    # === Display results ===
    print("\n🎯 Ensemble Metrics")
    print(f"📊 Mean Prediction:        {mean_pred:.4f}")
    print(f"📊 Median Prediction:      {median_pred:.4f}")
    print(f"📊 Trimmed Mean (middle):  {trimmed_mean:.4f}")
    print(f"📊 Winsorized Mean:        {winsor_mean:.4f}")
    print(f"📊 Weighted Average:       {weighted_avg:.4f}")
    predictions = {
        "mean_pred": round(mean_pred, 4),
        "med_pred": round(median_pred, 4),
        "trimmed_pred": round(trimmed_mean, 4),
        "Winsorized_pred": round(winsor_mean, 4),
        "weighted_avg": round(weighted_avg, 4),
    }

    # === Save results to CSV ===
    predictions_df = pd.DataFrame(predictions, index=[0])
    predictions_df.to_csv(
        f"/Users/lancesanterre/so_predict/data/predictions/{pitcher}_reg_predictions.csv",
        index=False,
    )
