"""
app.py

Streamlit app for interacting with the pitcher prediction pipeline. Users can:
- Add pitcher + opponent pairs for prediction
- Trigger predictions via FastAPI backend
- View saved regression and classification results

Author: Lance Santerre
"""

import streamlit as st
import requests
import pandas as pd
import time

st.title("🎯 Pitcher Prediction Dashboard")

# === Session state setup for storing pitcher-opponent pairs ===
if "pitcher_list" not in st.session_state:
    st.session_state.pitcher_list = []

# === Input section ===
st.subheader("Add Pitcher for Prediction")
pitcher_name = st.text_input("Pitcher's Full Name")
opponent = st.selectbox(
    "Opponent Team (Abbreviation)",
    [
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
        "WSN",
    ],
)

# === Add button logic ===
if st.button("Add Pitcher"):
    if pitcher_name and opponent:
        st.session_state.pitcher_list.append(
            (pitcher_name.strip(), opponent.strip().upper())
        )
        st.success(f"Added: {pitcher_name} vs {opponent}")
    else:
        st.warning("Please enter both a name and an opponent.")

# === Display current list of pitchers ===
if st.session_state.pitcher_list:
    st.subheader("Pitchers to Predict:")
    for i, (name, opp) in enumerate(st.session_state.pitcher_list):
        st.markdown(f"{i+1}. **{name}** vs **{opp}**")

# === Run the FastAPI prediction pipeline ===
if st.button("Run Predictions"):
    for name, opp in st.session_state.pitcher_list:
        with st.spinner(f"Running prediction for {name} vs {opp}..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/run_pipeline/",
                    json={"pitcher_name": name, "opponent": opp},
                    timeout=600,  # Increase timeout to 10 minutes for long scrapes
                )
                if response.status_code == 200:
                    st.success(f"Prediction complete for {name} vs {opp}.")
                else:
                    st.error(
                        f"Failed for {name} vs {opp}: {response.json().get('detail', 'Unknown error')}"
                    )
            except requests.exceptions.Timeout:
                st.error(f"Prediction for {name} vs {opp} timed out.")
            except Exception as e:
                st.error(f"Error running prediction for {name} vs {opp}: {e}")
    time.sleep(2)

# === View saved predictions from API ===
st.subheader("📂 View Saved Predictions")
response = requests.get("http://127.0.0.1:8000/predictions/")
if response.status_code == 200:
    prediction_names = response.json().get("available_predictions", [])
    if prediction_names:
        selected = st.selectbox("Choose prediction to view", prediction_names)
        if st.button("Load Prediction Data"):
            pred_data = requests.get(
                "http://127.0.0.1:8000/prediction_data/",
                params={"pitcher_opponent": selected},
            )
            if pred_data.status_code == 200:
                data = pred_data.json()
                st.subheader("Regression Predictions")
                st.dataframe(pd.DataFrame(data["regression"]))
                st.subheader("Classification Predictions")
                st.dataframe(pd.DataFrame(data["classification"]))
            else:
                st.error(
                    f"Failed to load prediction data: {pred_data.json().get('detail', 'Unknown error')}"
                )
    else:
        st.info("No predictions available yet.")
else:
    st.error("Failed to load prediction list.")
