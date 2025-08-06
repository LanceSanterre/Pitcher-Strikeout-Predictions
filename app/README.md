# === README.md ===
# SO Prediction App (Local)

This project uses **FastAPI** as the backend and **Streamlit** as the frontend.

## 🚀 How to Run

```bash
# In terminal 1: Start backend
cd so_predict_app/backend
uvicorn main:app --reload

# In terminal 2: Start frontend
cd so_predict_app/frontend
streamlit run app.py
```

Make sure your `run_pipeline()` saves two CSVs:
- `regression_predictions.csv`
- `classification_predictions.csv`

These should contain the `pitcher_id` column for lookup.

---

Place your original `.py` files (like `full_run.py`, `models.py`, etc.) into the `your_scripts/` folder and ensure they work locally.
