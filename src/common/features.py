"""Canonical feature definitions shared by training and inference."""

from __future__ import annotations

BASE_NUMERIC_FEATURES = [
    "days_until",
    "book_dow",
    "depart_dow",
    "depart_woy",
]

DERIVED_NUMERIC_FEATURES = [
    "booking_urgency",
    "days_until_squared",
    "is_major_holiday",
]

PROXY_NUMERIC_FEATURES = [
    "price_slope",
    "price_volatility_7d",
    "price_pct_change",
    "price_vs_min",
    "daily_price_points",
    "airline_count",
    "flight_count",
    "price_cv",
]

CATEGORICAL_FEATURES = [
    "route_O",
    "route_D",
    "holiday_category",
]

TARGET_COLUMN = "min_price"

XGBOOST_NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES + PROXY_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
)
XGBOOST_ALL_FEATURES = XGBOOST_NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASELINE_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES
BASELINE_CATEGORICAL_FEATURES = ["route_O", "route_D"]
BASELINE_ALL_FEATURES = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES

PROXY_FEATURE_LOOKUP_COLUMNS = [
    "route",
    "departure_date",
    "scraped_date",
    "scraped_time",
    TARGET_COLUMN,
    *PROXY_NUMERIC_FEATURES,
]


def route_code(origin: str, destination: str) -> str:
    """Return the canonical route label used throughout the repo."""
    return f"{origin.upper()}-{destination.upper()}"
