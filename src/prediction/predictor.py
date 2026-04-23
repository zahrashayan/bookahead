"""Canonical prediction service shared by future app/API callers."""

from __future__ import annotations

from datetime import date

from src.common.features import route_code
from src.prediction.feature_builder import build_prediction_features
from src.prediction.model_registry import load_model
from src.prediction.proxy_feature_service import history_row_count
from src.prediction.schemas import PredictionRequest, PredictionResult


def predict(request: PredictionRequest) -> PredictionResult:
    """Generate a route-level price prediction using historical proxy features."""
    booking_date = request.booking_date or date.today()
    features = build_prediction_features(request)
    model = load_model(request.origin, request.destination)
    predicted_price = float(model.predict(features)[0])

    return PredictionResult(
        route=route_code(request.origin, request.destination),
        predicted_price=round(predicted_price, 2),
        booking_date=booking_date,
        departure_date=request.departure_date,
        history_rows_used=history_row_count(
            request.origin,
            request.destination,
            request.departure_date,
            booking_date,
        ),
        feature_values=features.iloc[0].to_dict(),
    )
