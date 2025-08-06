"""
Module: pitcher_prediction_pipeline.py

Description:
This script serves as a full pipeline for predicting MLB pitcher performance.
It takes user input for pitcher names and opposing teams, scrapes relevant data
from multiple sources (Statcast, game logs, team-level data), engineers features,
and runs both regression and classification models to generate predictions.

Main Steps:
1. Accept pitcher and opponent input from user.
2. Scrape and extract data for the pitcher and team.
3. Construct a combined feature vector for model input.
4. Run predictions using trained regression and classification models.
5. Save results and print status updates to the console.

Usage:
Run the script and enter pitcher names and team abbreviations interactively.

Example:
$ python pitcher_prediction_pipeline.py
Pitcher Name (or 'done'): Blake Snell
Opponent Team Abbreviation: NYM

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
    Combine data from savant, team, and pitcher sources into a single row.

    Args:
        pitcher_id (str): The unique player ID for the pitcher.
        savant_data (DataFrame): DataFrame containing Statcast-style data.
        team_data (DataFrame): DataFrame containing team-level features.
        pitcher_data (DataFrame): DataFrame containing game log features.

    Returns:
        DataFrame: A single-row DataFrame with all features ordered and filled.
    """
    savant_row = (
        savant_data.iloc[0]
        if savant_data is not None and not savant_data.empty
        else pd.Series()
    )
    team_row = (
        team_data.iloc[0]
        if team_data is not None and not team_data.empty
        else pd.Series()
    )
    pitcher_row = (
        pitcher_data.iloc[0]
        if pitcher_data is not None and not pitcher_data.empty
        else pd.Series()
    )

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
            final_row[col] = 0  # Default

    return pd.DataFrame([final_row], columns=ordered_cols)


def run_pipeline(name, opp):
    """
    Full pipeline to run data scraping, preprocessing, feature building, and prediction.

    Args:
        name (str): Full name of the pitcher.
        opp (str): Opposing team abbreviation (e.g., "NYY").

    Returns:
        (bool, str): Success status and pitcher ID.
    """
    try:
        opp = opp.upper()
        clear_folder()
        pitcher_id = get_id(name)

        if not pitcher_id:
            print(f"❌ No ID found for {name}")
            return False, pitcher_id

        print(f"✅ Pitcher ID: {pitcher_id}")

        # Scrape data
        try:
            get_OPP_data(opp)
            get_player_data(pitcher_id)
        except Exception as e:
            print(f"❌ Failed to scrape data for {name} vs {opp}: {e}")
            return False, pitcher_id

        delete_team_file(opp)
        delete_year_log_file(pitcher_id, 2025)
        save_year_log_file(pitcher_id, 2025)
        save_year_log_file_team(opp)

        # Extract
        savant_data = extract_cols_savant(pitcher_id)
        team_data = extract_cols_team(opp, 2025)
        pitcher_data = extract_pitcher_recent_features(pitcher_id, 2025)

        if any(df is None or df.empty for df in [savant_data, team_data, pitcher_data]):
            print(f"❌ Missing data for {name} vs {opp}. Skipping.")
            return False, pitcher_id

        # Build final DF
        combined_df = build_combined_row(
            pitcher_id, savant_data, team_data, pitcher_data
        )
        combined_df.to_csv(
            "/Users/lancesanterre/so_predict/data/temp_data/combined_data.csv",
            index=False,
        )
        print("✅ Data saved to combined_data.csv")

        # Predictions
        print("🚀 Running predictions...")
        predict_pitcher_reg(pitcher_id)
        class_preds(pitcher_id)

        print(f"🎉 Prediction completed for {name} ({pitcher_id}) vs {opp}")
        return True, pitcher_id

    except Exception as e:
        print(f"❌ Pipeline crashed for {name} vs {opp}: {e}")
        return False, None


if __name__ == "__main__":
    pitcher_list = []
    print("\n🎯 Enter pitchers and opponents. Type 'done' to start predictions.\n")
    while True:
        name = input("Pitcher Name (or 'done'): ").strip()
        if name.lower() == "done":
            break
        opp = input("Opponent Team Abbreviation: ").strip().upper()
        while opp not in mlb_teams:
            print("❌ Invalid team abbreviation.")
            opp = input("Re-enter Opponent Team Abbreviation: ").strip().upper()
        pitcher_list.append((name, opp))
        print(f"Added: {name} vs {opp}")

    if not pitcher_list:
        print("🚫 No pitchers entered. Exiting.")
        exit()

    print("\n=== Starting predictions ===\n")
    for name, opp in pitcher_list:
        success, pid = run_pipeline(name, opp)
        if not success:
            print(f"⚠️ Skipped {name} ({opp}). Check logs for details.")
