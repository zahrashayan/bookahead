from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Flight Price Prediction API")

# Model path root
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Input schema
class FlightRequest(BaseModel):
    origin: str
    destination: str
    days_until_departure: int
    stops: int
    is_weekend: bool

@app.post("/predict")
def predict_price(data: FlightRequest):
    # Construct route key
    route_key = f"{data.origin}_{data.destination}"
    model_filename = f"xgb_{route_key}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # Check model existence
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model for route '{route_key}' not found.")

    try:
        # Load model pipeline (includes preprocessing)
        model = joblib.load(model_path)

        # Build feature row with correct column names
        features = pd.DataFrame([{
            "days_until": data.days_until_departure,
            "book_dow": 5,  # optional: dynamic logic to extract current day
            "depart_dow": 0,  # optional: or infer from other input if date given
            "depart_woy": 22,
            "route_O": data.origin,
            "route_D": data.destination,
        }])

        prediction = model.predict(features)[0]
        return {"predicted_price": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")




