from fastapi import FastAPI, HTTPException

from src.prediction.predictor import predict
from src.prediction.schemas import PredictionRequest

app = FastAPI(title="Flight Price Prediction API")

@app.post("/predict")
def predict_price(data: PredictionRequest):
    try:
        return predict(data).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
