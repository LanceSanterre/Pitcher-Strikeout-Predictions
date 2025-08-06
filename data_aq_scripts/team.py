"""
team.py

This script scrapes team-level batting statistics and rankings for all MLB teams listed in the `mlb_teams` list.
It uses Selenium and XPath to harvest tables from Baseball-Reference and saves them to individual CSV files.

Author: Lance Santerre
"""

from pitcher_scraper import PitcherGameLogScraper
import pandas as pd
import os
import random
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# List of MLB team acronyms
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
    "ARI",
]
# Expected column names for the main team batting table
colummns_names = [
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
]


class PitcherGameLogScraper:
    """
    Scrapes yearly batting stats and rankings for a given MLB team.

    Attributes:
        team_acro (str): 3-letter team acronym.
        save_root (str): Root path for saving CSV outputs.
    """

    def __init__(
        self,
        team_acro: str,
        save_root: str = "/Users/lancesanterre/so_predict/data/training/team_logs",
    ):
        self.team_acro = team_acro
        self.save_root = save_root
        self.driver = 0
        self.button = 0
        self.team_dir = os.path.join(save_root, team_acro)
        self.yby_rank = "//table[@id='yby_team_bat_ranks']/tbody"
        self.yby = '//table[@id="yby_team_bat"]/tbody'
        self.flag = False
        os.makedirs(self.team_dir, exist_ok=True)
        self.url = (
            f"https://www.baseball-reference.com/teams/{self.team_acro}/batteam.shtml"
        )

    def log_event(self, message: str):
        """Logs scraping events to a file for the current team."""
        base_log_dir = "/Users/lancesanterre/so_predict/data/training/logs"
        log_dir = os.path.join(base_log_dir, self.team_acro)
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "scrape_log.txt")
        with open(log_file, "a") as f:
            f.write(f"[{self.team_acro}] {message}\n")

    def take_some_time(self):
        """Sleeps for a random time between 15–30 seconds to simulate human browsing."""
        random_time = random.randint(15, 30)
        print("Sleeping for ", random_time, "seconds!")
        time.sleep(random_time)

    def harvest_data(self):
        """Scrapes one of the team batting tables depending on the value of self.flag."""
        print("Creating driver...")
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.driver.get(self.url)
        print("Driver Created ✅")

        self.take_some_time()
        if self.flag == False:

            print("It's Harvesting Time. Table 1..🌾 ")
            path = self.yby
            self.flag = True

        else:
            print("It's Harvesting Time. Table 2..🌾 ")
            path = self.yby_rank
        try:
            harvest = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, path))
            )
            data = harvest.text
        except Exception as e:
            msg = f"❌ Could not find the table: {e}"
            print(msg)
            self.log_event(msg)
            self.driver.quit()
            return None
        # Parse raw text into structured row data
        row_strings = data.strip().split("\n")
        row_data = []
        # Handle rows with unexpected column counts
        for row in row_strings:
            parts = row.split()
            new_row = []
            i = 0
            while i < len(parts):
                val = parts[i]
                new_row.append(val)
                i += 1

            row_data.append(new_row)

        expected_len = len(colummns_names)
        final_data = []

        for row in row_data:
            if len(row) == expected_len:
                final_data.append(row)
            elif len(row) == expected_len + 1:
                final_data.append(row[:expected_len] + [row[-1]])
            elif len(row) > expected_len + 1:
                self.log_event(f"Dropped row with length {len(row)} > 28: {row}")
            else:
                self.log_event(
                    f"⚠️ Skipping row with unexpected length {len(row)}: {row}"
                )

        columns = colummns_names.copy()
        if any(len(row) == expected_len + 1 for row in final_data):
            columns.append("vest")

        try:
            df = pd.DataFrame(final_data, columns=columns)
            print("✅ DataFrame created with", len(columns), "columns.")
            self.log_event("✅ Successfully scraped and parsed data.")
            return df
        except Exception as ve:
            msg = f"❌ Column mismatch error (final): {ve}"
            print(msg)
            self.log_event(msg)
            return None

    def save_csv(self, df: pd.DataFrame, table_name):
        """Saves a DataFrame to CSV if valid."""
        if df is None or not isinstance(df, pd.DataFrame):
            print("❌ Provided data is not a valid DataFrame. Skipping save.")
            return

        file_path = os.path.join(self.team_dir, f"{table_name}.csv")

        try:
            df.to_csv(file_path, index=False)
            print(f"✅ Saved to {file_path} with {df.shape[1]} columns.")
        except Exception as e:
            print(f"❌ Failed to save CSV for {self.team_dir}: {e}")

    def run(self):
        """Executes scraping of both tables and saves them to disk."""
        print("Let’s Start...")

        # ---- Table 1: yby ----
        self.flag = False
        df1 = self.harvest_data()
        self.driver.quit()
        if df1 is not None:
            self.save_csv(df1, "yby")
        else:
            print("❌ DataFrame was None — skipping save for yby.")

        # ---- Table 2: yby_rank ----
        print("\n🔁 Loading second table now...")
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.driver.get(self.url)
        self.take_some_time()
        self.flag = True  # now grab second table
        df2 = self.harvest_data()
        self.driver.quit()
        if df2 is not None:
            self.save_csv(df2, "yby_rank")
        else:
            print("❌ DataFrame was None — skipping save for yby_rank.")


# Main loop to scrape all MLB teams
for team_acro in mlb_teams:
    print(f"🚀 Processing: {team_acro}")
    scraper = PitcherGameLogScraper(team_acro)
    scraper.run()
