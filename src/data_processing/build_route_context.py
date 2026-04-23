"""Build a narrow route-context table for faster inference lookups."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.features import PROXY_FEATURE_LOOKUP_COLUMNS


INTERIM_PATH = Path(__file__).resolve().parents[2] / "data" / "interim" / "best_today.parquet"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "route_context_features.parquet"
)


def build_route_context_table() -> pd.DataFrame:
    """Create the inference-optimized subset used by history_repository."""
    if not INTERIM_PATH.exists():
        raise FileNotFoundError(f"Prepared panel not found: {INTERIM_PATH}")

    df = pd.read_parquet(INTERIM_PATH)
    missing_columns = [col for col in PROXY_FEATURE_LOOKUP_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Prepared panel is missing required route-context columns: {', '.join(missing_columns)}"
        )

    context = df[PROXY_FEATURE_LOOKUP_COLUMNS].copy()
    context = context.sort_values(["route", "departure_date", "scraped_date", "scraped_time"])
    context = context.drop_duplicates(
        subset=["route", "departure_date", "scraped_date", "scraped_time"],
        keep="last",
    )
    return context


def main():
    context = build_route_context_table()
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    context.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(context):,} route-context rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
