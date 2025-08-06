"""
load_team_data.py

This script reads all CSV files containing team game logs, stored in nested directories by team name.
It loads the logs, adds a 'team_name' column to each, and concatenates them into a single DataFrame.

Author: Lance Santerre
"""

import os
import pandas as pd


base_dir = "/Users/lancesanterre/so_predict/data/training/team_logs"
data_frames = []

for team in os.listdir(base_dir):
    team_path = os.path.join(base_dir, team)

    # ✅ Skip non-directory files like .DS_Store
    if not os.path.isdir(team_path):
        continue

    for file in os.listdir(team_path):
        full_path = os.path.join(team_path, file)
        try:
            df = pd.read_csv(full_path, header=None)
            df["team_name"] = team
            data_frames.append(df)
        except Exception as e:
            print(f"❌ Error reading {full_path}: {e}")

# Combine all game logs into one DataFrame
df_main = pd.concat(data_frames, ignore_index=True)

# Use the first row as the new column headers
df_main.columns = df_main.iloc[0]

# Drop the first row since it's now the header
df_main = df_main.drop(index=0).reset_index(drop=True)
df_main["team_name"] = df_main["PIT"]
df_main = df_main.drop(columns="PIT")

df_main["Lg"] = df_main["Lg"] + df_main["W"]

df_main = df_main.drop(columns="W")
df_main = df_main[df_main["Year"] != "Year"]


corrected_columns = [
    "Year",
    "Lg",
    "W",
    "L",
    "Finish",
    "R/G",
    "G",
    "PA",
    "AB",
    "R",
    "H",
    "2B",
    "3B",
    "HR",
    "RBI",
    "SB",
    "CS",
    "BB",
    "SO",
    "BA",
    "OBP",
    "SLG",
    "OPS",
    "E",
    "DP",
    "Fld%",
    "BatAge",
    "team_name",
]

# Truncate the dataframe to the correct number of columns
df_main = df_main.iloc[:, : len(corrected_columns)]

# Apply new column names
df_main.columns = corrected_columns

df_main.to_csv(
    "/Users/lancesanterre/so_predict/data/training/helper_data/cleaned_team_logs.csv",
    index=False,
)
print("✅ Columns realigned and saved!")
