"""
class_pred.py

This script loads all tree-based classification models (saved as .pkl), applies them to a specific pitcher's
features, and saves the ensemble prediction result (mode, average probability, hybrid score).

Author: Lance Santerre
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy import stats


def get_pitcher_name(
    pitcher_id,
    names_path="/Users/lancesanterre/so_predict/data/player_data/starting_pitchers.csv",
):
    """
    Look up a pitcher's name from the reference CSV. Used for naming output files.

    Args:
        pitcher_id (str): Unique identifier for the pitcher
        names_path (str): Path to pitcher ID-to-name reference file

    Returns:
        str: Pitcher's full name (underscored) or pitcher_id fallback
    """
    names_df = pd.read_csv(names_path)
    row = names_df[names_df["PlayerID"] == pitcher_id]
    if row.empty:
        return pitcher_id  # fallback to ID
    return row.iloc[0]["Name"].replace(" ", "_")


def class_preds(pitcher_id):
    """
    Generates ensemble classification prediction using all saved tree-based models for a given pitcher.

    Args:
        pitcher_id (str): Unique identifier of the pitcher (must match 'player_id' in input data)

    Saves:
        CSV file containing hybrid classification predictions
    """
    base_path = "/Users/lancesanterre/so_predict/models/class"
    input_path = "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv"

    df = pd.read_csv(input_path)
    input_row = df[df["player_id"] == pitcher_id]

    if input_row.empty:
        print(f"❌ Pitcher ID {pitcher_id} not found in input data.")
        return

    # Drop ID column before feeding into model
    input_row = input_row.drop(columns="player_id")

    probs, preds = [], []

    # Loop through all saved models in directory
    for file in os.listdir(base_path):
        if file.endswith("_model.pkl"):
            prefix = file.replace("_model.pkl", "")
            try:
                print(f"\n🌳 Processing Model: {prefix}")
                scaler = joblib.load(os.path.join(base_path, f"{prefix}_scaler.pkl"))
                model = joblib.load(os.path.join(base_path, f"{prefix}_model.pkl"))
                features = joblib.load(
                    os.path.join(base_path, f"{prefix}_features.pkl")
                )

                # Load continuous feature list
                if isinstance(features, dict):
                    cont_cols = features.get("cont_cols") or features.get("features")
                elif isinstance(features, list):
                    cont_cols = features
                else:
                    raise ValueError("Unsupported features format")

                input_data = input_row[cont_cols].dropna()
                X_scaled = scaler.transform(input_data)
                prob = model.predict_proba(X_scaled)[0][1]
                pred = int(prob > 0.4)
                probs.append(prob)
                preds.append(pred)
                print(f"✅ {prefix} → prob: {prob:.4f}, pred: {pred}")
            except Exception as e:
                print(f"❌ Error processing {prefix}: {e}")

    # === Aggregate predictions ===
    if preds:
        mode_class = int(stats.mode(preds, keepdims=True).mode[0])
        avg_prob = np.mean(probs)
        avg_class = int(avg_prob > 0.4)
        hybrid_score = 0.7 * mode_class + 0.3 * avg_class
        hybrid_class = int(hybrid_score >= 0.4)

        pitcher = get_pitcher_name(pitcher_id)

        summary = {
            "final_prediction": hybrid_class,
            "mode_class": mode_class,
            "avg_prob": round(avg_prob, 4),
            "hybrid_class": hybrid_class,
        }

        df_out = pd.DataFrame(summary, index=[0])
        out_path = (
            f"/Users/lancesanterre/so_predict/app/data/{pitcher}_class_predictions.csv"
        )
        df_out.to_csv(out_path, index=False)

        print("\n📊 Final Prediction Summary:")
        for k, v in summary.items():
            print(f"• {k}: {v}")
        print(f"✅ Saved to: {out_path}")
    else:
        print("⚠️ No valid predictions made.")
