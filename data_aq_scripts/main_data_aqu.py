"""
main_data_aqu.py

This script initiates the scraping of game log data for starting pitchers listed in a CSV file.
It reads pitcher data and uses the PitcherGameLogScraper to fetch logs across a range of years.

Author: Lance Santerre
"""

from pitcher_scraper import PitcherGameLogScraper
import pandas as pd


def get_years(start=1, end=100, pitcher_log="NA", inactive="NA"):
    """
    Determines how many years of data to scrape based on start and end year.

    Args:
        start (int): Start year for scraping.
        end (int): End year for scraping.
        pitcher_log (str): Pitcher name or ID for logging.
        inactive (str): String indicating inactive years (optional).

    Returns:
        int: Number of years to scrape or 0 if invalid input.
    """

    if start == 0 or pitcher_log == "NA":
        print(f"❌ ERROR: Data entered incorrectly for pitcher: {pitcher_log}")
        return 0
    elif start > end:
        print(
            f"⚠️ FINISHED: Start year {start} is greater than end year {end} for pitcher: {pitcher_log}"
        )
        return 0
    elif inactive == start:
        print(f"⏭️ Skipping inactive year {inactive} for {pitcher_log}")
        return get_years(start + 1, end, pitcher_log, inactive)

    print(f"📅 Scraping year {start} for {pitcher_log}")
    scraper = PitcherGameLogScraper(pitcher_log, start)

    if scraper.passover():
        print("✅ Year data has already been logged")
    else:
        try:
            scraper.run()
        except Exception as e:
            print(f"❌ Error for {pitcher_log} in {start}: {e}")

    return get_years(start + 1, end, pitcher_log, inactive)


# Load pitcher data
pitchers = pd.read_csv(
    "starting_pitchers.csv"
)

# Loop through all pitchers and run the scraper
for index, row in pitchers.iterrows():
    pitcher_log = row[1]
    start_year = row[-3]
    end_year = row[-2]
    inactive = row[-1]

    print(f"🚀 Processing: {pitcher_log} ({start_year} - {end_year})")
    get_years(
        start=start_year, end=end_year, pitcher_log=pitcher_log, inactive=inactive
    )
