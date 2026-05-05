"""Booking recommendation logic built on prediction and route context."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.prediction.history_repository import load_route_context_table
from src.prediction.schemas import PredictionResult


@dataclass(frozen=True)
class BookingRecommendation:
    """User-facing booking guidance derived from model context."""

    recommendation: str
    status: str
    explanation: str
    confidence: str


def make_booking_recommendation(result: PredictionResult) -> BookingRecommendation:
    """Return transparent booking guidance from prediction metadata."""
    features = result.feature_values
    route = result.route
    days_until = int(features["days_until"])
    gap_to_best = float(features["price_vs_min"])
    price_slope = float(features["price_slope"])
    pct_change = float(features["price_pct_change"])
    volatility = float(features["price_volatility_7d"])
    history_rows = int(result.history_rows_used)
    thresholds = get_route_recommendation_thresholds(route)

    if history_rows < 5:
        return BookingRecommendation(
            recommendation="MONITOR PRICES",
            status="info",
            explanation=(
                "There is limited route history for this departure date, so the safest "
                "recommendation is to keep monitoring before making a decision."
            ),
            confidence="Low",
        )

    if days_until <= 14:
        return BookingRecommendation(
            recommendation="BOOK NOW",
            status="error",
            explanation=(
                "The departure date is close, and short booking windows usually leave "
                "less room for prices to recover."
            ),
            confidence=_confidence(history_rows, volatility),
        )

    if gap_to_best <= thresholds["gap_low"] and (
        price_slope >= thresholds["slope_flat"]
        or pct_change >= thresholds["pct_flat"]
    ):
        return BookingRecommendation(
            recommendation="GOOD TIME TO BOOK",
            status="success",
            explanation=(
                "The current prediction is close to the best observed price for this "
                "route context, and recent movement is not clearly improving."
            ),
            confidence=_confidence(history_rows, volatility),
        )

    if price_slope >= thresholds["slope_upward"] or pct_change >= thresholds["pct_upward"]:
        return BookingRecommendation(
            recommendation="BOOK SOON",
            status="warning",
            explanation=(
                "Recent route context suggests upward pressure, so waiting may increase "
                "the risk of paying more."
            ),
            confidence=_confidence(history_rows, volatility),
        )

    if days_until > 60 and volatility >= thresholds["volatility_high"]:
        return BookingRecommendation(
            recommendation="MONITOR PRICES",
            status="info",
            explanation=(
                "The trip is still far out and recent prices have been volatile, so "
                "monitoring gives the model more context before committing."
            ),
            confidence=_confidence(history_rows, volatility),
        )

    if gap_to_best <= thresholds["gap_ok"] and 21 <= days_until <= 60:
        return BookingRecommendation(
            recommendation="GOOD TIME TO BOOK",
            status="success",
            explanation=(
                "The prediction is within a reasonable range of the best observed price "
                "and the departure is inside a practical booking window."
            ),
            confidence=_confidence(history_rows, volatility),
        )

    return BookingRecommendation(
        recommendation="MONITOR PRICES",
        status="info",
        explanation=(
            "The current signal is not urgent. Continue monitoring for a better price "
            "or stronger upward momentum."
        ),
        confidence=_confidence(history_rows, volatility),
    )


def _confidence(history_rows: int, volatility: float) -> str:
    """Simple confidence label based on history depth and volatility."""
    if history_rows >= 30 and volatility < 20:
        return "High"
    if history_rows >= 10:
        return "Medium"
    return "Low"


@lru_cache(maxsize=32)
def get_route_recommendation_thresholds(route: str) -> dict:
    """Calibrate recommendation thresholds from route history distributions."""
    df = load_route_context_table()
    route_df = df[df["route"] == route]

    if route_df.empty:
        return {
            "gap_low": 15.0,
            "gap_ok": 40.0,
            "slope_flat": 0.0,
            "slope_upward": 10.0,
            "pct_flat": 0.0,
            "pct_upward": 0.03,
            "volatility_high": 20.0,
        }

    def q(series_name: str, quantile: float, floor: float | None = None) -> float:
        value = float(route_df[series_name].quantile(quantile))
        if floor is not None:
            value = max(floor, value)
        return value

    return {
        "gap_low": q("price_vs_min", 0.25, 5.0),
        "gap_ok": q("price_vs_min", 0.5, 15.0),
        "slope_flat": float(np.median(route_df["price_slope"])),
        "slope_upward": q("price_slope", 0.75, 1.0),
        "pct_flat": float(np.median(route_df["price_pct_change"])),
        "pct_upward": q("price_pct_change", 0.75, 0.01),
        "volatility_high": q("price_volatility_7d", 0.75, 10.0),
    }
