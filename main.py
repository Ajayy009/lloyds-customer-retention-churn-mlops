from fastapi import FastAPI
import pandas as pd
import joblib
import mlflow
import os

app = FastAPI()

# 1. Setup Direct Connection to the Shared Volume
MLFLOW_TRACKING_URI = "sqlite:///mlflow_data/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2. Loading Logic
try:
    # Explicitly using the name we see in your screenshots
    model_uri = "runs:/9400a78ea61c44ef95f5e882d182e406/model"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print(f"SUCCESS: Loaded {model_uri} from shared volume!")
    model_source = "MLflow Registry (Shared Volume)"
except Exception as e:
    print(f"MLflow Access failed ({e}). Falling back to local pickle...")
    model = joblib.load("model.pkl")
    model_source = "Local Pickle (Backup)"

@app.get("/")
def home():
    return {"status": "Online", "mode": "Shared-Volume-Access", "brain_source": model_source}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {
        "prediction": "Churn" if prediction == 1 else "Stay",
        "model_used": model_source
    }