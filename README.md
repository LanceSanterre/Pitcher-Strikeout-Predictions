# MLB Starting Pitcher Strikeout Predictions

This project predicts strikeouts for MLB starting pitchers using machine learning models, combined with live web scraping to keep data fresh and relevant. The models are trained on historical game logs and advanced statistics pulled from sources like Baseball Reference and MLB Savant.

---

## 🧠 Overview

- **Goal**: Use machine learning to predict the number of strikeouts for each starting pitcher in MLB games.
- **Approach**: Gather real-time data using web scraping, process and clean it, then feed it to multiple ML models for prediction.
- **Inspiration**: Exploring the power of data in sports forecasting and building a pipeline that is both functional and informative.

---

## 🛠️ How to Run

### 1. Scrape Data
Run the scraping scripts to gather up-to-date data:
- `pitcher_scraper.py` – Scrapes individual pitcher game logs.
- `team.py` – Scrapes team-level stats.

### 2. Combine Data
- Use `data_combining.py` to merge pitcher, team, and historical data.

### 3. Train Models (optional if using pre-trained models)
- Classification: `model_classification_training.py`
- Regression: `model_regression_training.py`

### 4. Run Predictions
Use `full_run.py` or run individual scripts:
- `regression_pred.py`
- `class_pred.py`

**Input Needed:**
- Pitcher ID (e.g., `colege01` for Gerrit Cole)
- Team Abbreviation (e.g., `NYY` for Yankees)

---

## 📁 Repository Notes

To keep the repo clean and secure, the following items are **excluded via `.gitignore`**:
- All local datasets (`data/`)
- Saved model files (`models/`)
- MLflow tracking logs (`mlruns/`)
- Artifacts (`mlartifacts/`)
- CSVs and pickled objects (`*.csv`, `*.pkl`)
- Jupyter checkpoint and system files (`__pycache__/`, `.DS_Store`, `*.ipynb_checkpoints/`)

If you're cloning this project, you’ll need to run the scraping and training scripts to rebuild the full environment.

---

## 📊 Data Sources

- [Baseball Reference](https://www.baseball-reference.com)
- [MLB Savant](https://baseballsavant.mlb.com)

---

## 👨‍💻 Author

**Lance Santerre**
