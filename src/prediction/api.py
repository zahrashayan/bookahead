from fastapi import FastAPI, HTTPException

from src.prediction.predictor import predict
from src.prediction.recommendation import make_booking_recommendation
from src.prediction.schemas import PredictionAPIResponse, PredictionRequest

app = FastAPI(title="Flight Price Prediction API")

@app.post("/predict", response_model=PredictionAPIResponse)
def predict_price(data: PredictionRequest):
    try:
        result = predict(data)
        recommendation = make_booking_recommendation(result)
        return PredictionAPIResponse(
            **result.model_dump(),
            recommendation=recommendation.recommendation,
            recommendation_status=recommendation.status,
            recommendation_explanation=recommendation.explanation,
            recommendation_confidence=recommendation.confidence,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
