"""Canonical inference-time feature assembly."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.common.features import XGBOOST_ALL_FEATURES
from src.prediction.proxy_feature_service import build_proxy_features
from src.prediction.schemas import PredictionRequest


MAJOR_HOLIDAYS = {
    "thanksgiving": ["11-24", "11-25", "11-26", "11-27"],
    "christmas": ["12-23", "12-24", "12-25", "12-26", "12-27"],
    "new_years": ["12-30", "12-31", "01-01", "01-02"],
    "spring_break": ["03-15", "03-22"],
    "july_4th": ["07-03", "07-04", "07-05"],
    "labor_day": ["09-01", "09-02", "09-03"],
    "memorial_day": ["05-25", "05-26", "05-27"],
}


def get_holiday_category(value: date) -> str:
    """Map a date to the current coarse holiday buckets used in training."""
    month_day = value.strftime("%m-%d")
    for holiday_name, dates in MAJOR_HOLIDAYS.items():
        if month_day in dates:
            return holiday_name
    return "none"


def build_prediction_features(request: PredictionRequest) -> pd.DataFrame:
    """Build a one-row feature dataframe that matches the XGBoost training contract."""
    booking_date = request.booking_date or date.today()
    departure_ts = pd.Timestamp(request.departure_date)
    booking_ts = pd.Timestamp(booking_date)
    days_until = int((departure_ts - booking_ts).days)

    if days_until < 0:
        raise ValueError("booking_date cannot be after departure_date")

    holiday_category = get_holiday_category(request.departure_date)
    proxy_features = build_proxy_features(
        request.origin,
        request.destination,
        request.departure_date,
        booking_date,
    )

    row = {
        "days_until": days_until,
        "book_dow": int(booking_ts.dayofweek),
        "depart_dow": int(departure_ts.dayofweek),
        "depart_woy": int(departure_ts.isocalendar().week),
        "booking_urgency": int(days_until < 14),
        "days_until_squared": int(days_until ** 2),
        "is_major_holiday": int(holiday_category != "none"),
        "route_O": request.origin.upper(),
        "route_D": request.destination.upper(),
        "holiday_category": holiday_category,
        **proxy_features,
    }

    return pd.DataFrame([[row[column] for column in XGBOOST_ALL_FEATURES]], columns=XGBOOST_ALL_FEATURES)
