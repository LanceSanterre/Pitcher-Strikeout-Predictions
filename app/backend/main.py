"""
main.py

A FastAPI application that exposes three endpoints:
1. /run_pipeline/ - Accepts pitcher name and opponent, runs the full prediction pipeline
2. /predictions/ - Lists all available completed predictions
3. /prediction_data/ - Returns regression and classification data for a given pitcher

Author: Lance Santerre
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scrips")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from full_run import run_pipeline  # Uses updated pipeline (Savant + Team + Rolling)

# Add scripts directory to system path
PREDICTION_DIR = "/Users/lancesanterre/so_predict/app/data"

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PitcherInput(BaseModel):
    """Defines the expected input format for a POST request to /run_pipeline/"""
    pitcher_name: str
    opponent: str


@app.post("/run_pipeline/")
def run_prediction(input_data: PitcherInput):
    """
    Trigger the full prediction pipeline for a given pitcher and opponent.

    Args:
        input_data (PitcherInput): Contains pitcher name and opponent team

    Returns:
        JSON message confirming pipeline execution and pitcher ID
    """
    name = input_data.pitcher_name.strip()
    opp = input_data.opponent.strip().upper()

    try:
        success, pid = run_pipeline(name, opp)
        if not success:
            raise HTTPException(
                status_code=500, detail=f"Pipeline failed for {name} vs {opp}"
            )
        return {
            "message": f"Prediction completed for {name} vs {opp}",
            "pitcher_id": pid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline crashed: {str(e)}")


@app.get("/predictions/")
def list_predictions():
    """
    Lists all pitcher IDs that have both regression and classification prediction files.

    Returns:
        JSON list of available prediction keys (pitcher IDs)
    """
    preds = []
    for filename in os.listdir(PREDICTION_DIR):
        if filename.endswith("_class_predictions.csv"):
            pitcher = filename.replace("_class_predictions.csv", "")
            reg_file = os.path.join(PREDICTION_DIR, f"{pitcher}_reg_predictions.csv")
            if os.path.exists(reg_file):
                preds.append(pitcher)
    return {"available_predictions": preds}


@app.get("/prediction_data/")
def load_prediction(pitcher_opponent: str):
    """
    Loads regression and classification predictions for a specific pitcher.

    Args:
        pitcher_opponent (str): Pitcher ID string (usually name_opponent)

    Returns:
        JSON object containing both prediction types
    """
    name = pitcher_opponent.strip()
    reg_path = os.path.join(PREDICTION_DIR, f"{name}_reg_predictions.csv")
    class_path = os.path.join(PREDICTION_DIR, f"{name}_class_predictions.csv")

    if not (os.path.exists(reg_path) and os.path.exists(class_path)):
        raise HTTPException(status_code=404, detail="Prediction files not found.")

    reg_df = pd.read_csv(reg_path)
    class_df = pd.read_csv(class_path)

    return {
        "regression": reg_df.to_dict(orient="records"),
        "classification": class_df.to_dict(orient="records"),
    }
