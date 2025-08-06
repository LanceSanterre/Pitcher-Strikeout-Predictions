"""
full_run.py

This script builds a full data pipeline for pitcher prediction.
It:
- Extracts data for a given pitcher and opponent
- Combines savant, pitcher, and team-level stats
- Saves a combined CSV
- Runs regression and classification predictions
- Accepts interactive or test-mode inputs

Author: Lance Santerre
"""

import pandas as pd
from helper import *
from regression_pred import *
from class_pred import *

mlb_teams = [
    "WSN",
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
]


def build_combined_row(pitcher_id, savant_data, team_data, pitcher_data):
    """
    Combines pitcher, team, and savant data into a single structured row for predictions.

    Args:
        pitcher_id (str): The ID of the pitcher
        savant_data (DataFrame): Extracted Savant features
        team_data (DataFrame): Team statistics
        pitcher_data (DataFrame): Recent game log statistics

    Returns:
        pd.DataFrame: A one-row dataframe of combined features
    """

    savant_row = savant_data.iloc[0] if not savant_data.empty else pd.Series()
    team_row = team_data.iloc[0] if not team_data.empty else pd.Series()
    pitcher_row = pitcher_data.iloc[0] if not pitcher_data.empty else pd.Series()

    # Final ordered columns
    ordered_cols = [
        "Gtm",
        "player_id",
        "pitches",
        "total_pitches",
        "pitch_percent",
        "ba",
        "iso",
        "babip",
        "slg",
        "woba",
        "xwoba",
        "xba",
        "hits",
        "abs",
        "launch_speed",
        "launch_angle",
        "spin_rate",
        "velocity",
        "effective_speed",
        "whiffs",
        "swings",
        "takes",
        "eff_min_vel",
        "release_extension",
        "pos3_int_start_distance",
        "pos4_int_start_distance",
        "pos5_int_start_distance",
        "pos6_int_start_distance",
        "pos7_int_start_distance",
        "pos8_int_start_distance",
        "pos9_int_start_distance",
        "pitcher_run_exp",
        "run_exp",
        "bat_speed",
        "swing_length",
        "pa",
        "bip",
        "singles",
        "doubles",
        "triples",
        "hrs",
        "so_1",
        "k_percent",
        "bb_1",
        "bb_percent",
        "api_break_z_with_gravity",
        "api_break_z_induced",
        "api_break_x_arm",
        "api_break_x_batter_in",
        "hyper_speed",
        "bbdist",
        "hardhit_percent",
        "barrels_per_bbe_percent",
        "barrels_per_pa_percent",
        "release_pos_z",
        "release_pos_x",
        "plate_x",
        "plate_z",
        "obp",
        "barrels_total",
        "batter_run_value_per_100",
        "xobp",
        "xslg",
        "pitcher_run_value_per_100",
        "xbadiff",
        "xobpdiff",
        "xslgdiff",
        "wobadiff",
        "swing_miss_percent",
        "arm_angle",
        "attack_angle",
        "attack_direction",
        "swing_path_tilt",
        "rate_ideal_attack_angle",
        "intercept_ball_minus_batter_pos_x_inches",
        "intercept_ball_minus_batter_pos_y_inches",
        "handed",
        "season",
        "avg_IP_5",
        "avg_H_5",
        "avg_R_5",
        "avg_ER_5",
        "avg_HR_5",
        "avg_BB_5",
        "avg_IBB_5",
        "avg_SO_5",
        "avg_HBP_5",
        "avg_BK_5",
        "avg_WP_5",
        "avg_BF_5",
        "avg_ERA_5",
        "avg_FIP_5",
        "avg_Pit_5",
        "avg_Str_5",
        "avg_StL_5",
        "avg_StS_5",
        "avg_GB_5",
        "avg_FB_5",
        "avg_LD_5",
        "avg_PU_5",
        "avg_Unk_5",
        "avg_GmSc_5",
        "avg_SB_5",
        "avg_CS_5",
        "avg_AB_5",
        "avg_2B_5",
        "avg_GIDP_5",
        "avg_ROE_5",
        "avg_aLI_5",
        "avg_WPA_5",
        "avg_acLI_5",
        "Year",
        "W",
        "L",
        "Finish",
        "R/G",
        "PA_1",
        "R",
        "RBI",
        "CS",
        "SO_2",
        "BA_1",
        "OBP_1",
        "SLG_1",
        "OPS",
        "E",
        "DP",
        "Fld%",
        "BatAge",
        "OPP_SO_PRE",
    ]

    # Build row as dict
    final_row = {}
    for col in ordered_cols:
        if col == "player_id":
            final_row[col] = pitcher_id
        elif col in savant_row.index:
            final_row[col] = savant_row[col]
        elif col in team_row.index:
            final_row[col] = team_row[col]
        elif col in pitcher_row.index:
            final_row[col] = pitcher_row[col]
        else:
            final_row[col] = 0  # Default if missing

    # Return as single-row DataFrame
    return pd.DataFrame([final_row], columns=ordered_cols)


def run_pipeline(name, opp):
    """
    Executes full pipeline for one pitcher:
    - Acquires all data
    - Combines into row
    - Saves row
    - Runs both regression and classification predictions

    Args:
        name (str): Pitcher's full name
        opp (str): Opponent team abbreviation

    Returns:
        tuple: (bool, pitcher_id)
    """
    opp = opp.upper()
    clear_folder()
    pitcher_id = get_id(name)

    if pitcher_id is None:
        print(f"❌ No ID found for {name}")
        return False, pitcher_id

    print(f"✅ Pitcher ID: {pitcher_id}")

    # === Data acquisition ===
    get_OPP_data(opp)
    get_player_data(pitcher_id)
    delete_team_file(opp)
    delete_year_log_file(pitcher_id, 2025)
    save_year_log_file(pitcher_id, 2025)
    save_year_log_file_team(opp)

    # Debugging
    savant_data = extract_cols_savant(pitcher_id)

    team_data = extract_cols_team(opp, 2025)

    pitcher_data = extract_pitcher_recent_features(pitcher_id, 2025)
    # Merge everything
    combined_df = build_combined_row(pitcher_id, savant_data, team_data, pitcher_data)
    print(combined_df)
    combined_df.to_csv(
        "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv", index=False
    )
    print("✅ Data saved to combined_data.csv")

    # === Predictions ===
    print("🚀 Running predictions...")
    predict_pitcher_reg(pitcher_id)
    class_preds(pitcher_id)

    return True, pitcher_id


# === Run Main ===
if __name__ == "__main__":
    # === Collect Inputs ===
    pitcher_list = []

    # For testing: set to True to skip manual input
    test_mode = True
    if test_mode:
        pitcher_list = [("Logan Webb", "LAD"), ("Max Scherzer", "BOS")]
    else:
        print("🎯 Enter pitcher names and opponent teams. Type 'done' when finished.\n")
        while True:
            name = input(
                "🔍 Enter the pitcher's full name (or 'done' to finish): "
            ).strip()
            if name.lower() == "done":
                break

            opp = (
                input("🏟️  Enter opponent team abbreviation (e.g., BAL): ")
                .strip()
                .upper()
            )
            while opp not in mlb_teams:
                print(
                    "❌ Invalid team abbreviation. Please enter a valid MLB team (e.g., BAL, NYY, LAD)."
                )
                opp = input("🏟️  Re-enter opponent team abbreviation: ").strip().upper()

            pitcher_list.append((name, opp))

    if not pitcher_list:
        print("🚫 No inputs provided. Exiting.")
        exit()

    # === Validation Step ===
    print("\n📋 Validating all pitchers before starting...\n")
    validated_pitchers = []

    for name, opp in pitcher_list:
        pitcher_id = get_id(name)
        print(pitcher_id)
        if not pitcher_id:  # Only skip if None or empty
            print(f"❌ Could not find a valid pitcher ID for {name} — skipping.")
        else:
            validated_pitchers.append((name, opp, pitcher_id))
            print(f"✅ {name} ({pitcher_id})")

    if not validated_pitchers:
        print("\n🚫 No valid pitchers found. Exiting.")
        exit()

    # === Processing Each Pitcher One-by-One ===
    print("\n✅ Starting data acquisition and prediction for each pitcher...\n")
    for name, opp, pitcher_id in validated_pitchers:
        try:
            print(f"\n⚾️ Processing {name} vs {opp}...\n")
            success, pid = run_pipeline(name, opp)
            if not success:
                print(f"❌ Failed for {name} ({pid}) — skipping.\n")
        except Exception as e:
            print(f"⚠️ Error processing {name} ({pitcher_id}): {e}")
