"""Shared request/response schemas for prediction."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Business-level prediction inputs provided by the caller."""

    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    departure_date: date
    booking_date: Optional[date] = None


class PredictionResult(BaseModel):
    """Structured prediction response."""

    route: str
    predicted_price: float
    booking_date: date
    departure_date: date
    history_rows_used: int
    feature_values: dict
