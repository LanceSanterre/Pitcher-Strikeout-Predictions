"""
regression_pred.py

This script:
- Loads pitcher-specific features
- Applies scaling using a pre-fit scaler
- Runs predictions using Random Forest and XGBoost models
- Averages results
- Saves predictions to a CSV file named after the pitcher

Author: Lance Santerre
"""

import os
import joblib
import pandas as pd
import numpy as np

# Paths
model_dir = "/Users/lancesanterre/so_predict/models/reg"
data_path = "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv"
output_dir = "/Users/lancesanterre/so_predict/app/data"

os.makedirs(output_dir, exist_ok=True)


# Helper: Get pitcher name
def get_pitcher_name(
    pitcher_id,
    names_path="/Users/lancesanterre/so_predict/data/player_data/starting_pitchers.csv",
):
    """
    Look up the full pitcher name from a reference CSV using the pitcher ID.

    Args:
        pitcher_id (str): Baseball-Reference-style pitcher ID.
        names_path (str): Path to the CSV containing ID-to-name mappings.

    Returns:
        str: Pitcher's name with underscores instead of spaces (or pitcher_id fallback).
    """
    names_df = pd.read_csv(names_path)
    row = names_df[names_df["PlayerID"] == pitcher_id]
    if row.empty:
        return pitcher_id  # fallback to ID
    return row.iloc[0]["Name"].replace(" ", "_")


def predict_pitcher_reg(pitcher_id):
    """
    Generate regression predictions for a specific pitcher using trained models.

    Loads features and scaler, extracts the row matching the pitcher,
    applies scaling, and computes predictions using both RF and XGB models.

    Saves the result to a CSV file under the pitcher’s name.

    Args:
        pitcher_id (str): Unique pitcher ID (e.g., "degroja01").

    Raises:
        ValueError: If pitcher ID is not found in the input data.
    """
    # Load shared scaler & features
    features = joblib.load(os.path.join(model_dir, "features.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    # Load combined data & select pitcher
    df = pd.read_csv(data_path)
    input_row = df[df["player_id"] == pitcher_id]
    if input_row.empty:
        raise ValueError(f"❌ Pitcher ID {pitcher_id} not found in combined data.")
    input_row = input_row[features]  # Select correct fields

    # Scale
    input_scaled = scaler.transform(input_row)

    # Load models
    models = {
        "rf": joblib.load(os.path.join(model_dir, "rf_model.pkl")),
        "xgb": joblib.load(os.path.join(model_dir, "xgb_model.pkl")),
    }

    # Predictions
    results = {name: model.predict(input_scaled)[0] for name, model in models.items()}
    results["average_prediction"] = np.mean(list(results.values()))

    # Save to CSV
    pitcher_name = get_pitcher_name(pitcher_id)
    output_path = os.path.join(output_dir, f"{pitcher_name}_reg_predictions.csv")
    pd.DataFrame([results]).to_csv(output_path, index=False)
    print(f"✅ Predictions for {pitcher_name} saved to {output_path}")
