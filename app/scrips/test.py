"""
test.py

A test script that allows me to see my prediction work

Author: Lance Santerre
"""
import joblib
import pandas as pd
import numpy as np
import os

# === Paths ===
base_path = "/Users/lancesanterre/so_predict/models/class/"
feature_file = "deep_adam_robust_20250722_132946_features.pkl"
output_path = "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv"

# === Load Feature Info ===
feature_info = joblib.load(os.path.join(base_path, feature_file))
cont_cols = feature_info["cont_cols"]

# === Print Expected Columns ===
print("✅ The model expects the following continuous columns:\n")
for col in cont_cols:
    print(f"• {col}")

# === Create Dummy Input ===
dummy_input = pd.DataFrame([np.random.rand(len(cont_cols))], columns=cont_cols)

print("\n📄 Sample Input Row:")
print(dummy_input)

# === Save to CSV ===
dummy_input.to_csv(output_path, index=False)
print(f"\n📁 Saved dummy input to: {output_path}")
