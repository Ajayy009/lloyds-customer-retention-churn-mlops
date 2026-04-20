from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# 1. Load the model directly from the file we just saved
# This makes it independent of MLflow
try:
    model = joblib.load("model.pkl")
    print("SUCCESS: Lloyds Model is now LIVE via local pickle file!")
except Exception as e:
    print(f"ERROR LOADING MODEL: {e}")

@app.get("/")
def home():
    return {"status": "Online", "mode": "Docker-Ready"}

@app.post("/predict")
def predict(data: dict):
    # Convert incoming JSON data to a DataFrame for the model
    df = pd.DataFrame([data])
    
    # Generate prediction
    prediction = model.predict(df)[0]
    
    return {
        "prediction": "Churn" if prediction == 1 else "Stay",
        "model_version": "Tuned_RF_Local"
    }