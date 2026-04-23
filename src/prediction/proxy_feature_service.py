"""Expose historical proxy features for inference."""

from __future__ import annotations

from datetime import date

from src.common.features import PROXY_NUMERIC_FEATURES
from src.prediction.history_repository import (
    get_latest_available_context_row,
    get_route_departure_history,
)


def build_proxy_features(
    origin: str,
    destination: str,
    departure_date: date,
    booking_date: date,
) -> dict:
    """Return proxy features derived from the latest available route context row."""
    row = get_latest_available_context_row(origin, destination, departure_date, booking_date)
    return {feature: row[feature] for feature in PROXY_NUMERIC_FEATURES}


def history_row_count(
    origin: str,
    destination: str,
    departure_date: date,
    booking_date: date,
) -> int:
    """Return the number of historical rows contributing to the context window."""
    history = get_route_departure_history(origin, destination, departure_date, booking_date)
    return int(len(history))
