"""
Module: player_scraper.py

Description:
Scrapes MLB pitcher game logs from Baseball Reference using Selenium and LXML, 
then parses, logs, and stores the data as CSV files. This is part of the 
pitcher analytics pipeline used for training machine learning models on individual pitcher performance.

Key Features:
- Navigates to a specific pitcher's yearly game log page.
- Uses Selenium to click the 'wide table' tooltip for full stats.
- Parses table rows into structured format using robust row handling logic.
- Logs errors, skips malformed rows, and stores successful scrapes.
- Saves the structured data as a CSV file.

Main Class:
- PitcherGameLogScraper

Dependencies:
- Selenium
- LXML
- Pandas
- OS / Time / Random / Datetime

Author: Lance Santerre
"""

import os
from lxml import html
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


colummns_names = [
    "Rk",
    "Gcar",
    "Gtm",
    "Date",
    "Team",
    "Opp",
    "Result",
    "Inngs",
    "Dec",
    "DR",
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
    "PO",
    "AB",
    "2B",
    "3B",
    "GIDP",
    "SF",
    "ROE",
    "aLI",
    "WPA",
    "acLI",
    "cWPA",
]


class PitcherGameLogScraper:
    """
    Scraper class to extract MLB pitcher game logs from Baseball-Reference
    using Selenium and store them as CSV files. It handles navigation, table
    interaction, parsing, logging, and saving of data.

    Attributes:
        pitcher_id (str): Unique Baseball-Reference ID for the pitcher.
        year (int): The season year to scrape.
        save_root (str): Directory to save the scraped data.
    """

    def __init__(
        self,
        pitcher_id: str,
        year: int,
        save_root: str = "/Users/lancesanterre/so_predict/data/temp_data",
    ):
        self.pitcher_id = pitcher_id
        self.year = year
        self.save_root = save_root
        self.driver = 0
        self.button = 0
        self.pitcher_dir = os.path.join(save_root, pitcher_id)
        os.makedirs(self.pitcher_dir, exist_ok=True)
        self.url = f"https://www.baseball-reference.com/players/gl.fcgi?id={self.pitcher_id}&t=p&year={self.year}"

    def log_event(self, message: str):
        """
        Append a message to the log file associated with this pitcher and year.

        Args:
            message (str): The message to log.
        """
        # Base log directory (absolute path)
        base_log_dir = "/Users/lancesanterre/so_predict/data/training/logs"
        log_dir = os.path.join(base_log_dir, self.pitcher_id)
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "scrape_log.txt")
        with open(log_file, "a") as f:
            f.write(f"[{self.year}] {message}\n")

    def take_some_time(self):
        """
        Randomly sleep between 15 and 30 seconds to simulate human-like delay
        during scraping and reduce likelihood of being blocked.
        """
        random_time = random.randint(15, 30)
        print("Sleeping for ", random_time, "seconds!")
        time.sleep(random_time)

    def harvest_data(self):
        """
        Main method to scrape and parse the pitcher’s game logs table from the webpage.

        Returns:
            pd.DataFrame or None: Parsed game log data as a DataFrame, or None if failed.
        """
        print("Creating driver...")
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.driver.get(self.url)
        print("Driver Created ✅")

        self.take_some_time()

        print("Scrolling and waiting for wide table button...")
        try:
            self.button = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//li[@class="scroll_note"]//span[@class="tooltip"]')
                )
            )
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                self.button,
            )
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//li[@class="scroll_note"]//span[@class="tooltip"]')
                )
            )
            self.take_some_time()
            self.driver.execute_script("arguments[0].click();", self.button)
            print("Clicked ✅")
        except Exception as e:
            msg = f"❌ Could not click wide table button: {e}"
            print(msg)
            self.log_event(msg)
            self.driver.quit()
            return None

        print("It's Harvesting Time...🌾 ")
        try:
            harvest = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//table[@id="players_standard_pitching"]/tbody')
                )
            )
            data = harvest.text
        except Exception as e:
            msg = f"❌ Could not find the table: {e}"
            print(msg)
            self.log_event(msg)
            self.driver.quit()
            return None

        row_strings = data.strip().split("\n")
        row_data = []
        for row in row_strings:
            parts = row.split()
            new_row = []
            i = 0
            while i < len(parts):
                val = parts[i]
                if val == "@":
                    i += 1
                    continue
                if "," in val:
                    new_row.append(val + " " + parts[i + 1])
                    i += 2
                elif val.startswith("("):
                    new_row[-1] += " " + val
                    i += 1
                elif val.startswith("GS-"):
                    new_row.append(val)
                    if i + 1 < len(parts) and (
                        parts[i + 1].startswith(("W(", "L(", "H("))
                    ):
                        new_row.append(parts[i + 1])
                        i += 2
                    else:
                        new_row.append("NA")
                        i += 1
                elif "%" in val:
                    new_row.append(val)
                    i += 1
                    break
                else:
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
                self.log_event(f"Dropped row with length {len(row)} > 48: {row}")
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

    def save_csv(self, df: pd.DataFrame):
        """
        Save the scraped game log DataFrame as a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing game logs.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            print("❌ Provided data is not a valid DataFrame. Skipping save.")
            return

        file_path = os.path.join(self.pitcher_dir, f"{self.year}_log.csv")

        try:
            df.to_csv(file_path, index=False)
            print(f"✅ Saved to {file_path} with {df.shape[1]} columns.")
        except Exception as e:
            print(f"❌ Failed to save CSV for {self.pitcher_id} in {self.year}: {e}")

    def passover(self):
        """
        Check if a game log CSV file for this pitcher and year already exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(self.pitcher_dir, f"{self.year}_log.csv")
        return os.path.isfile(file_path)

    def run(self):
        """
        Orchestrates the scraping pipeline:
        - Harvests the data
        - Closes the Selenium driver
        - Saves the data if successful
        """
        print("Lets Start...")
        df = self.harvest_data()
        self.driver.quit()
        if df is not None:
            self.save_csv(df)
        else:
            print("❌ DataFrame was None — skipping save.")
