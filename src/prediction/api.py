from fastapi import FastAPI, HTTPException

from src.prediction.predictor import predict
from src.prediction.recommendation import make_booking_recommendation
from src.prediction.schemas import PredictionRequest

app = FastAPI(title="Flight Price Prediction API")

@app.post("/predict")
def predict_price(data: PredictionRequest):
    try:
        result = predict(data)
        recommendation = make_booking_recommendation(result)
        payload = result.model_dump()
        payload["recommendation"] = recommendation.recommendation
        payload["recommendation_status"] = recommendation.status
        payload["recommendation_explanation"] = recommendation.explanation
        payload["recommendation_confidence"] = recommendation.confidence
        return payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
