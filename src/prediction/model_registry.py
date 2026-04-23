"""Shared model loading and route->artifact resolution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def model_path_for_route(origin: str, destination: str, prefix: str = "xgb") -> Path:
    """Return the canonical model artifact path for a route."""
    filename = f"{prefix}_{origin.upper()}_{destination.upper()}.pkl"
    return MODELS_DIR / filename


@lru_cache(maxsize=32)
def load_model(origin: str, destination: str, prefix: str = "xgb"):
    """Load and cache a model artifact."""
    model_path = model_path_for_route(origin, destination, prefix=prefix)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)
