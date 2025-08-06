MLB Starting Pitcher Strikeout Predictions
==========================================

This project predicts strikeouts for MLB starting pitchers using machine learning, combined with live web scraping to keep data current. Models are trained on historical pitcher game logs and advanced statistics from Baseball Reference and MLB Savant.

Project Overview
----------------
- **Goal:** Predict how many strikeouts each MLB starting pitcher will have in today's games.
- **Approach:**
  - Scrape up-to-date data
  - Combine pitcher, team, and advanced stat features
  - Feed into ML classification and regression models
- **Inspiration:** Explore the intersection of sports forecasting and machine learning by building a reproducible, end-to-end pipeline.

How to Run
----------
With the included `starting_pitcher.csv` file, you can now run everything by executing:

    python data_aq_scripts/main_data_aqu.py

This script automates:
- Scraping pitcher logs
- Scraping team logs
- Combining all data
- Running classification & regression predictions

Requirements
------------
- pandas
- selenium
- scikit-learn
- ChromeDriver (installed and available in your PATH)

Input Format
------------
The `starting_pitcher.csv` file should contain:

    player_id,team_acronym
    colege01,NYY

This is used to scrape the correct data and generate predictions.

Key Folders and Scripts
-----------------------
- `data_aq_scripts/` – Scraping and data combination scripts
  - `main_data_aqu.py` – Entry point
  - `pitcher_scraper.py` – Scrapes pitcher logs
  - `team.py` – Scrapes opponent team data
  - `data_combining.py` – Combines all data
- `training_scripts/` – ML training and prediction
  - `model_regression_training.py`
  - `model_classification_training.py`
  - `regression_pred.py`
  - `class_pred.py`
- `app/` – Streamlit and FastAPI app interface
- `.gitignore` – Ensures `data/`, `models/`, and `.pkl`/`.csv` files are not tracked
- `README.md` – This file

Data Sources
------------
- https://www.baseball-reference.com
- https://baseballsavant.mlb.com

Author
------
Lance Santerre  
Built as a demonstration of real-world sports ML pipelines.  
Happy to walk through any part of the codebase or discuss further improvements.
