"""
class_pred.py

Loads and applies multiple trained classification models (PyTorch-based) to a single pitcher data row.
The script aggregates predictions using mode, average probability, and a hybrid voting strategy.
Final predictions are saved as a CSV.

Author: Lance Santerre
"""

from collections import Counter
import os
import torch
import joblib
import pandas as pd
from models import DeepModel_class, TransformerModel
import numpy as np
from scipy import stats


def class_preds(pitcher):
    """
    Run ensemble classification predictions for a given pitcher using saved neural network models.

    Args:
        pitcher (str): Unique pitcher identifier used to name output CSV

    Saves:
        CSV file with final prediction values in the predictions directory
    """
    base_path = "/Users/lancesanterre/so_predict/models/class"
    input_path = "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv"

    input_df = pd.read_csv(input_path)
    model_files = [f for f in os.listdir(base_path) if f.endswith("_model.pt")]

    probs = []  # list of predicted probabilities
    preds = []  # list of binary class predictions

    for model_file in model_files:
        prefix = model_file.replace("_model.pt", "")
        print(f"\n🔍 Processing: {prefix}")

        try:
            # Load preprocessing tools and model metadata
            scaler = joblib.load(f"{base_path}/{prefix}_scaler.pkl")
            feature_info = joblib.load(f"{base_path}/{prefix}_features.pkl")
            metadata = joblib.load(f"{base_path}/{prefix}_metadata.pkl")

            cont_cols = feature_info["cont_cols"]
            cat_cols = feature_info["cat_cols"]

            # Prepare input data for model
            X_cont = torch.tensor(
                scaler.transform(input_df[cont_cols]), dtype=torch.float32
            )
            X_cat = torch.tensor(input_df[cat_cols].values, dtype=torch.long).view(
                -1, len(cat_cols)
            )

            # Dynamically select model class
            if prefix.startswith("TransformerModel"):
                model = TransformerModel(
                    n_cont=metadata["num_cont_features"],
                    n_cat=metadata["embedding_input_dim"],
                    emb_dim=metadata["embedding_output_dim"],
                    output_type="classification",
                )
            else:
                # Make prediction
                model = DeepModel_class(
                    n_cont=metadata["num_cont_features"],
                    n_cat=metadata["embedding_input_dim"],
                    emb_dim=metadata["embedding_output_dim"],
                    output_type="classification",
                )

            model.load_state_dict(torch.load(f"{base_path}/{prefix}_model.pt"))
            model.eval()

            with torch.no_grad():
                prob = model(X_cont, X_cat).item()
                pred = int(prob > 0.5)
                probs.append(prob)
                preds.append(pred)

                print(f"✅ {prefix} → prob: {prob:.4f}, pred: {pred}")

        except Exception as e:
            print(f"❌ Error processing {prefix}: {e}")

    # ---- Aggregate and Save Predictions ----
    if preds:
        mode_class = int(stats.mode(preds, keepdims=True).mode[0])

        # 2. Average Probability
        avg_prob = np.mean(probs)
        avg_class = int(avg_prob > 0.5)

        # 3. Hybrid Score (optional)
        # Weighted average of mode and average prob prediction
        # (e.g., 70% weight to mode, 30% to prob-based)
        hybrid_score = 0.7 * mode_class + 0.3 * avg_class
        hybrid_class = int(hybrid_score >= 0.5)

        # 🎯 Print Final Summary
        print("\n📊 Final Prediction Summary:")
        print(f"🧮 Mode of predicted classes: {mode_class}")
        print(f"📈 Average probability: {avg_prob:.4f} → Class: {avg_class}")
        print(f"🔀 Hybrid class (70% mode, 30% prob): {hybrid_class}")
        # Save predictions to CSV
        predictions = {
            "final_prediction": hybrid_class,
            "mode_class": mode_class,
            "avg_prob": round(avg_prob, 4),
            "hybrid_class": hybrid_class,
        }
        predictions_df = pd.DataFrame(predictions, index=[0])
        predictions_df.to_csv(
            f"/Users/lancesanterre/so_predict/data/predictions/{pitcher}_class_predictions.csv",
            index=False,
        )

    else:
        print("⚠️ No predictions were made.")
