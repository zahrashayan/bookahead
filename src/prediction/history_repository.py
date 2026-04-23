"""Lookup helpers for historical route context used at inference time."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.common.features import route_code


INTERIM_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "interim" / "best_today.parquet"


@lru_cache(maxsize=1)
def load_route_context_table() -> pd.DataFrame:
    """Load the prepared panel once and normalize key datetime fields."""
    if not INTERIM_DATA_PATH.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {INTERIM_DATA_PATH}")

    df = pd.read_parquet(INTERIM_DATA_PATH).copy()
    df["scraped_date"] = pd.to_datetime(df["scraped_date"])
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    df["route"] = df["route"].astype(str)
    df["scraped_time"] = df["scraped_time"].astype(str)
    return df


def get_route_departure_history(
    origin: str,
    destination: str,
    departure_date: date,
    booking_date: date,
) -> pd.DataFrame:
    """Return historical rows available as of booking_date for a route/departure pair."""
    df = load_route_context_table()
    route = route_code(origin, destination)
    departure_ts = pd.Timestamp(departure_date)
    booking_ts = pd.Timestamp(booking_date)

    history = df[
        (df["route"] == route)
        & (df["departure_date"] == departure_ts)
        & (df["scraped_date"] <= booking_ts)
    ].copy()

    if history.empty:
        return history

    return history.sort_values(["scraped_date", "scraped_time"])


def get_latest_available_context_row(
    origin: str,
    destination: str,
    departure_date: date,
    booking_date: date,
) -> pd.Series:
    """Return the latest historical row known as of booking_date."""
    history = get_route_departure_history(origin, destination, departure_date, booking_date)
    if history.empty:
        raise ValueError(
            "No historical context available for "
            f"{route_code(origin, destination)} on {departure_date} as of {booking_date}"
        )
    return history.iloc[-1]
