"""
data_combining.py

This script traverses a directory containing game log CSV files for individual pitchers,
converts the 'IP' (innings pitched) values to decimal format, and combines all the logs
into a single CSV file.

Author: Lance Santerre
"""

import os
import pandas as pd
from datetime import datetime


def fix_ip(ip):
    """
        Converts innings pitched (IP) from baseball-style notation to decimal format.

    Baseball uses:
    - .1 = 1 out = 1/3 inning
    - .2 = 2 outs = 2/3 inning

    Args:
        ip (float or NaN): The original IP value (e.g., 5.1, 6.2)

    Returns:
        float or None: Converted decimal IP (e.g., 5.1 -> 5.333...), or None if input is NaN
    """

    if pd.isna(ip):
        return None
    whole = int(ip)
    decimal = round(ip - whole, 1)
    # 0.1 = 1 out, 0.2 = 2 outs
    if decimal == 0.1:
        return whole + 1 / 3
    elif decimal == 0.2:
        return whole + 2 / 3
    return whole


# Directory where individual pitcher logs are stored
base_dir = "/Users/lancesanterre/so_predict/data/training/game_logs"
data_frames = []

# Loop through each pitcher directory
for pitcher_name in os.listdir(base_dir):
    pitcher_path = os.path.join(base_dir, pitcher_name)

    # Loop through each file in the pitcher's folder
    for year_file in os.listdir(pitcher_path):
        full_path = os.path.join(pitcher_path, year_file)

        try:
            df = pd.read_csv(full_path, header=None)
            df["pitcher_id"] = pitcher_name
            data_frames.append(df)
        except Exception as e:
            print(f"❌ Error reading {full_path}: {e}")

# Combine all game logs into one DataFrame
df_main = pd.concat(data_frames, ignore_index=True)

# Use the first row as header
df_main.columns = df_main.iloc[0]
df_main = df_main[1:].reset_index(drop=True)
df_main = df_main.rename(columns={"snellbl01": "pitcher_id"})
# Transformations
transformations = df_main.drop(columns=["Rk"])
transformations["Gcar"] = pd.to_numeric(transformations["Gcar"], errors="coerce")
transformations["Gtm"] = pd.to_numeric(transformations["Gtm"], errors="coerce")
transformations["Date"] = pd.to_datetime(transformations["Date"], errors="coerce")
# Clean and standardize team names
print("Team, Opp Fixing")
# Master mapping: wrong → correct
team_replacements = {
    "TBR": "TB",
    "TB": "TB",
    "TBD": "TB",
    "Rays": "TB",
    "Bay": "TB",
    "SFG": "SFG",
    "SF": "SFG",
    "Giants": "SFG",
    "LAD": "LAD",
    "Dodgers": "LAD",
    "SD": "SDP",
    "SDP": "SDP",
    "Padres": "SDP",
    "KCR": "KC",
    "KC ": "KC",
    "Royals": "KC",
    "WSN": "WSH",
    "Nationals": "WSH",
    "FLA": "MIA",
    "Marlins": "MIA",
    "NYM": "NYM",
    "Mets": "NYM",
    "NYY": "NYY",
    "Yankees": "NYY",
    "York": "NYY",
    "CHW": "CHW",
    "CWS": "CHW",
    "CHC": "CHC",
    "Cubs": "CHC",
    "DET": "DET",
    "Tigers": "DET",
    "MIL": "MIL",
    "Brewers": "MIL",
    "LAA": "LAA",
    "Angels": "LAA",
    "Angeles": "LAA",
    "TEX": "TEX",
    "Rangers": "TEX",
    "OAK": "OAK",
    "Athletics": "OAK",
    "ATH": "OAK",
    "MON": "NA",  # Expos - defunct
    "Team": "NA",
    "Opp": "NA",
    "to": "NA",  # header artifacts or invalids
}

# Apply to both Team and Opp columns
for col in ["Team", "Opp"]:
    transformations[col] = transformations[col].astype(str).str.strip()
    transformations[col] = transformations[col].replace(team_replacements)

# Drop rows where standardization failed (still not in the 30-team list)
mlb_teams = [
    "ARI",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CHW",
    "CIN",
    "CLE",
    "COL",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "OAK",
    "PHI",
    "PIT",
    "SDP",
    "SEA",
    "SFG",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSH",
]

transformations = transformations[
    transformations["Team"].isin(mlb_teams) & transformations["Opp"].isin(mlb_teams)
].copy()
transformations["cWPA"] = transformations["cWPA"].str.replace("%", "").astype(float)

# Now map to IDs
team_to_id = {team: idx for idx, team in enumerate(sorted(mlb_teams))}
transformations["Team_ID"] = transformations["Team"].map(team_to_id)
transformations["Opp_ID"] = transformations["Opp"].map(team_to_id)

print("Team, Opp FIXED and no errors")


transformations["IP"] = transformations["IP"].astype(float).apply(fix_ip)


# Extract first character from Dec column
transformations["Decision"] = transformations["Dec"].astype(str).str[0].str.upper()

# Replace only known values, treat the rest as no decision
transformations["Decision"] = (
    transformations["Decision"].map({"W": 1, "L": -1}).fillna(0).astype(int)
)

transformations["DR"] = pd.to_numeric(transformations["DR"], errors="coerce")
transformations["Start_Depth"] = (
    transformations["Inngs"].str.extract(r"GS-(\d+)").astype(float)
)


# Ensure the required columns are numeric
transformations["SO"] = pd.to_numeric(transformations["SO"], errors="coerce")
transformations["BB"] = pd.to_numeric(transformations["BB"], errors="coerce")
transformations["IP"] = pd.to_numeric(transformations["IP"], errors="coerce")

# Rolling averages per pitcher
transformations = transformations.sort_values(by=["pitcher_id", "Date"])
transformations["Rolling_SO_5"] = transformations.groupby("pitcher_id")["SO"].transform(
    lambda x: x.shift().rolling(5).mean()
)
transformations["Rolling_BB_5"] = transformations.groupby("pitcher_id")["BB"].transform(
    lambda x: x.shift().rolling(5).mean()
)
transformations["Rolling_IP_5"] = transformations.groupby("pitcher_id")["IP"].transform(
    lambda x: x.shift().rolling(5).mean()
)

# Final features for training
final_features = [
    "DR",
    "Start_Depth",
    "Team_ID",
    "Opp_ID",
    "Rolling_SO_5",
    "Rolling_BB_5",
    "Rolling_IP_5",
    "pitcher_id",
    "BB",
    "Date",
]

# Convert DR and Start_Depth to numeric if needed
transformations["DR"] = pd.to_numeric(transformations["DR"], errors="coerce")
transformations["Start_Depth"] = pd.to_numeric(
    transformations["Start_Depth"], errors="coerce"
)

# Drop rows with missing values
model_data = transformations[final_features + ["SO"]].dropna()


today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"/Users/lancesanterre/so_predict/data/training/helper_data/model_data_{today_str}.parquet"
# Save to Parquet with today's date in the filename
model_data.to_parquet(filename, index=False)

print(f"✅ Saved Parquet file as: {filename}")
