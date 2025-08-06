"""
combine_predictions.py

This script reads individual regression and classification prediction CSV files for each pitcher,
prefixes their columns with the prediction type, and combines them into a single CSV for analysis.

Author: Lance Santerre
"""

import os
import pandas as pd
from datetime import date

today = date.today()


def combine_reg_class_predictions(folder_path):
    """
    Combines all individual regression and classification predictions in a folder into a single DataFrame.

    Args:
        folder_path (str): Path to folder containing *_reg_predictions.csv and *_class_predictions.csv files

    Returns:
        pd.DataFrame: Combined prediction DataFrame with prefixed column names
    """
    records = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder_path, filename)

        # Identify file type and unique name
        if "_reg_pred" in filename:
            unique_name = filename.replace("_reg_predictions.csv", "")
            kind = "reg"
        elif "_class_pred" in filename:
            unique_name = filename.replace("_class_predictions.csv", "")
            kind = "class"
        else:
            continue  # Skip unrelated files

        try:
            df = pd.read_csv(filepath)

            if df.shape[0] > 1:
                print(f"⚠️ Warning: {filename} has more than one row; using the first.")

            row_data = df.iloc[0].to_dict()

            # Initialize row if not already
            if unique_name not in records:
                records[unique_name] = {}

            # Prefix columns with type (reg/class)
            prefixed = {f"{key}_{kind}": value for key, value in row_data.items()}
            records[unique_name].update(prefixed)

        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")

    # Convert dict to DataFrame
    df_combined = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df_combined.rename(columns={"index": "unique_name"}, inplace=True)

    return df_combined


# === Run the combiner and export ===
combined_df = combine_reg_class_predictions("/Users/lancesanterre/so_predict/app/data")
combined_df.to_csv(
    f"/Users/lancesanterre/so_predict/data/combined_pred/{today}_pitcher_preds_combined.csv",
    index=False,
)
