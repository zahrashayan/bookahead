"""
Simple starter for data collection.
Run: python src/data_collection.py
"""
import pandas as pd
from datetime import datetime
import os

def save_sample_csv(path="data/raw/sample_prices.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([
        {"route":"SFO-JFK", "date_collected": datetime.today().isoformat(), "departure_date":"2025-10-15", "price":420},
        {"route":"SFO-LHR", "date_collected": datetime.today().isoformat(), "departure_date":"2025-11-01", "price":850},
    ])
    df.to_csv(path, index=False)
    print(f"Saved sample to {path}")

if __name__ == "__main__":
    save_sample_csv()
