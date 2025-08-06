"""
helper.py

This script provides utility functions for:
- Scraping player and team data
- File cleanup and saving
- Feature extraction (rolling averages, opponent metrics, savant stats)
- Resolving player names to IDs using DuckDB
- Loading saved predictions

Author: Lance Santerre
"""

import duckdb
import pandas as pd
from team_scrap import TeamGameLogScraper
from player_scrap import PitcherGameLogScraper
import os
import shutil


def normalize_player_id(df):
    """
    Normalize any column referring to player ID to be named 'player_id'.

    Args:
        df (pd.DataFrame): Input dataframe with potentially inconsistent ID column names.

    Returns:
        pd.DataFrame: DataFrame with 'player_id' as the standardized column name.
    """
    if df is None or df.empty:
        return df
    for col in df.columns:
        if col.lower() in ["playerid", "player_id", "pitcherid", "pitcher_id"]:
            df = df.rename(columns={col: "player_id"})
            break
    return df


def repair_combined_row(pitcher_id, savant_data, team_data, pitcher_data):
    """
    Rebuild a combined row when duplicate player_id issues arise.

    Args:
        pitcher_id (str): The ID of the pitcher.
        savant_data (DataFrame): Savant metrics data.
        team_data (DataFrame): Team-level statistics.
        pitcher_data (DataFrame): Pitcher game log data.

    Returns:
        pd.DataFrame: Combined and repaired row for the pitcher.
    """
    print(f"⚠️ Repairing row for {pitcher_id} (duplicate player_id found).")

    # Drop player_id columns from inputs
    for df in [savant_data, team_data, pitcher_data]:
        if df is not None and not df.empty and "player_id" in df.columns:
            df.drop(columns=["player_id"], inplace=True)

    # Rebuild using the standard method
    return build_combined_row(pitcher_id, savant_data, team_data, pitcher_data)


def get_id(name):
    """
    Retrieve the player ID based on the player's name.

    Args:
        name (str): Full name of the player.

    Returns:
        str: Player ID or error message if not found.
    """
    # Load CSVs
    id_names = pd.read_csv(
        "/Users/lancesanterre/so_predict/data/player_data/starting_pitchers.csv"
    )
    clusters = pd.read_csv(
        "/Users/lancesanterre/so_predict/data/training/helper_data/cleaned_full.csv"
    )
    name = name.lower()
    # Connect to DuckDB
    con = duckdb.connect()
    con.register("id_names", id_names)
    con.register("clusters", clusters)

    # Step 1: Get pitcher ID
    pitcher_id_query = f"""
        SELECT PlayerID
        FROM id_names
        WHERE LOWER(Name) = '{name}'
        LIMIT 1
    """
    pitcher_id_df = con.execute(pitcher_id_query).fetchdf()

    if pitcher_id_df.empty:
        return (
            f"❌ No player found for '{name}'. Please check the spelling and try again.",
            None,
        )

    pitcher_id = pitcher_id_df.iloc[0]["PlayerID"]

    return pitcher_id


def get_OPP_data(opp):
    """Scrape opponent team data for a given abbreviation."""
    OPP = opp.upper()
    scraper = TeamGameLogScraper(OPP)
    scraper.run()


def get_player_data(id):
    """Scrape pitcher game log data for the 2025 season."""
    scraper = PitcherGameLogScraper(id, 2025)
    scraper.run()


def clear_folder():
    """
    Clears all files in the /temp_data folder.
    """
    folder_path = "/Users/lancesanterre/so_predict/data/temp_data"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}: {e}")

    print(f"✅ Folder cleared: {folder_path}")


def delete_team_file(opp):
    """
    Delete the most recent team log file for a given team abbreviation.
    """
    base_dir = f"/Users/lancesanterre/so_predict/data/training/team_logs/{opp}"
    log_filename = "yby.csv"
    target_path = os.path.join(base_dir, log_filename)

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            print(f"🗑️ Deleted: {log_filename} from {opp}")
        except Exception as e:
            print(f"⚠️ Failed to delete {log_filename} from {opp}: {e}")
    else:
        print(f"❌ File does not exist: {target_path}")


def save_year_log_file_team(opp):
    """
    Save team log from temp directory to team_logs directory.
    """
    base_dir = f"/Users/lancesanterre/so_predict/data/training/team_logs/{opp}"
    os.makedirs(base_dir, exist_ok=True)  # Ensure team directory exists
    df = pd.read_csv(f"/Users/lancesanterre/so_predict/data/temp_data/{opp}/yby.csv")
    df.to_csv(os.path.join(base_dir, "yby.csv"), index=False)
    print(f"Data Saved to {base_dir}/yby.csv!")


def delete_year_log_file(id, year):
    """
    Delete a specific pitcher's game log file by year.
    """
    base_dir = "/Users/lancesanterre/so_predict/data/training/game_logs"
    log_filename = f"{year}_log.csv"
    target_path = os.path.join(base_dir, id, log_filename)

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            print(f"🗑️ Deleted: {log_filename} from {id}")
        except Exception as e:
            print(f"⚠️ Failed to delete {log_filename} from {id}: {e}")
    else:
        print(f"❌ File does not exist: {target_path}")


def save_year_log_file(id, year):
    """
    Save team log from temp directory to team_logs directory.
    """
    base_dir = os.path.join(
        "/Users/lancesanterre/so_predict/data/training/game_logs", id
    )
    os.makedirs(base_dir, exist_ok=True)  # Ensure pitcher directory exists
    df = pd.read_csv(
        f"/Users/lancesanterre/so_predict/data/temp_data/{id}/{year}_log.csv"
    )
    df.to_csv(os.path.join(base_dir, f"{year}_log.csv"), index=False)
    print(f"Data Saved to {base_dir}/{year}_log.csv!")


def extract_cols_team(opp, year):
    """
    Delete a specific pitcher's game log file by year.
    """
    path = f"/Users/lancesanterre/so_predict/data/training/team_logs/{opp}/yby.csv"
    data = pd.read_csv(path)
    data = normalize_player_id(data)

    if "Lg" in data.columns and "W" in data.columns:
        data["Lg"] = data["Lg"].astype(str) + " " + data["W"].astype(str)
        cols = list(data.columns)
        for i in range(2, len(cols) - 1):
            data[cols[i]] = data[cols[i + 1]]
        data = data.iloc[:, :-1]

    data = data.drop(
        columns=[col for col in data.columns if "Unnamed" in col], errors="ignore"
    )
    data = data[data["Year"].apply(lambda x: str(x).isdigit())].copy()
    data["Year"] = data["Year"].astype(int)
    data_saved = data[data["Year"] == year].copy()

    if data_saved.empty:
        print(f"⚠️ No data found for team {opp} in year {year}, filling with zeros.")
        columns = [
            "BatAge",
            "SB",
            "DP",
            "2B",
            "3B",
            "E",
            "CS",
            "BB",
            "AB",
            "HR",
            "OPP_SO_PRE",
        ]
        return pd.DataFrame([{col: 0 for col in columns}])

    if "SO" in data_saved.columns and "PA" in data_saved.columns:
        data_saved["OPP_SO_PRE"] = data_saved["SO"].astype(float) / data_saved[
            "PA"
        ].astype(float)
    else:
        data_saved["OPP_SO_PRE"] = 0.0

    keep_cols = [
        "BatAge",
        "SB",
        "DP",
        "2B",
        "3B",
        "E",
        "CS",
        "BB",
        "AB",
        "HR",
        "OPP_SO_PRE",
    ]
    for col in keep_cols:
        if col not in data_saved.columns:
            data_saved[col] = 0

    return data_saved[keep_cols]


def extract_pitcher_recent_features(id, year):
    df = pd.read_csv(
        f"/Users/lancesanterre/so_predict/data/training/game_logs/{id}/{year}_log.csv"
    )
    df = normalize_player_id(df)
    full_df = pd.read_csv(
        "/Users/lancesanterre/so_predict/data/training/helper_data/cleaned_full.csv"
    )
    full_df = normalize_player_id(full_df)

    cluster_row = full_df[full_df["player_id"] == id]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    rolling_cols = [
        "IP",
        "H",
        "R",
        "ER",
        "HR",
        "BB",
        "IBB",
        "SO",
        "HBP",
        "BK",
        "WP",
        "BF",
        "ERA",
        "FIP",
        "Pit",
        "Str",
        "StL",
        "StS",
        "GB",
        "FB",
        "LD",
        "PU",
        "Unk",
        "GmSc",
        "SB",
        "CS",
        "AB",
        "2B",
        "GIDP",
        "ROE",
        "aLI",
        "WPA",
        "acLI",
    ]
    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rolling = df[rolling_cols].rolling(window=5).mean()
    for col in rolling_cols:
        df[f"avg_{col}_5"] = rolling[col]

    df["days_since_last_pitch"] = df["Date"].diff().dt.days.fillna(5)

    if df.dropna(subset=[f"avg_{col}_5" for col in rolling_cols]).empty:
        print(f"⚠️ Not enough games for rolling averages for {id}, filling with zeros.")
        result = {f"avg_{col}_5": 0 for col in rolling_cols}
    else:
        last_row = df.dropna(subset=[f"avg_{col}_5" for col in rolling_cols]).iloc[-1]
        result = {f"avg_{col}_5": last_row[f"avg_{col}_5"] for col in rolling_cols}

    result["days_since_last_pitch"] = df["days_since_last_pitch"].iloc[-1]

    return pd.DataFrame([result])


PRED_DIR = "/Users/lancesanterre/so_predict/data/predictions"


def load_predictions(pitcher_name):
    """
    Loads classification and regression predictions for a given pitcher from CSVs.
    Returns:
        class_df: pd.DataFrame of classification predictions
        reg_df: pd.DataFrame of regression predictions
    """
    class_path = os.path.join(PRED_DIR, f"{pitcher_name}_class_predictions.csv")
    reg_path = os.path.join(PRED_DIR, f"{pitcher_name}_reg_predictions.csv")

    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Missing file: {class_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Missing file: {reg_path}")

    class_df = pd.read_csv(class_path)
    reg_df = pd.read_csv(reg_path)

    return class_df, reg_df


def extract_cols_savant(pitcher_id):
    path = "/Users/lancesanterre/so_predict/data/player_data/savant_with_ids.csv"
    data = pd.read_csv(path)
    data = normalize_player_id(data)
    data = data[data["player_id"] == pitcher_id]
    numeric_cols = data.select_dtypes(include=["number"]).columns
    mean_values = data[numeric_cols].mean()
    if data.empty:
        # Fill with zero columns if missing
        columns = [
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
        ]
        print(
            f"⚠️ No Savant data found for pitcher_id: {pitcher_id}. Filling with dataset averages."
        )
        # Create a single-row DataFrame with means
        pitcher_data = pd.DataFrame([mean_values])
        pitcher_data.insert(0, "player_id", pitcher_id)
        # Fill any non-numeric columns with default values (e.g., 0 or 'unknown')
        for col in data.columns:
            if col not in pitcher_data.columns:
                pitcher_data[col] = 0
        data.insert(0, "player_id", pitcher_id)

    return data
