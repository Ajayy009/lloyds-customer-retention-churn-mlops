from fastapi import FastAPI
import pandas as pd
import pickle
import os

app = FastAPI()

# Load XGBoost model from pickle
try:
    with open("best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("SUCCESS: Loaded XGBoost model from pickle!")
    model_source = "XGBoost Pickle"
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    model_source = "Failed"

THRESHOLD = 0.3  # Same threshold you used in training

@app.get("/")
def home():
    return {
        "status": "Online",
        "model": "XGBoost",
        "model_source": model_source
    }

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= THRESHOLD)
    return {
        "prediction": "Churn" if prediction == 1 else "Stay",
        "churn_probability": round(float(prob), 4),
        "threshold_used": THRESHOLD,
        "model_used": model_source
    }