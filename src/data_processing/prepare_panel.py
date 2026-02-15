# src/ml/prepare_panel.py
# builds a clean dataset (panel) from all raw scraped flight data

import os
import pandas as pd
import numpy as np

# paths to raw csv and the cleaned parquet file
RAW = os.path.join(os.path.dirname(__file__), '../../data/raw/flights_data_cleaned.csv')
INTERIM = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')

def main():
    print("Loading raw flight-level data...")
    df = pd.read_csv(RAW, encoding='utf-8')

    # convert string columns to datetime / numeric where needed
    df['scraped_date']   = pd.to_datetime(df['scraped_date'])
    df['scraped_time']   = df['scraped_time'].astype(str)
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    # remove extreme values (sometimes scraping glitches)
    df = df[(df['price'] >= 30) & (df['price'] <= 5000)]

    # group by each scrape snapshot so we keep one min price per route/date/time
    best = (
        df.groupby(['route','departure_date','scraped_date','scraped_time'], as_index=False)
          .agg(min_price=('price','min'))
    )

    # create extra features for the ML model
    best['days_until'] = (best['departure_date'] - best['scraped_date']).dt.days
    best = best[best['days_until'] >= 0]
    best['book_dow']   = best['scraped_date'].dt.dayofweek
    best['depart_dow'] = best['departure_date'].dt.dayofweek
    best['depart_woy'] = best['departure_date'].dt.isocalendar().week.astype(int)
    best[['route_O','route_D']] = best['route'].str.split('-', expand=True)

    # unique id for each record (so later we can split chronologically)
    best['panel'] = (
        best['route'] + '|' +
        best['departure_date'].astype(str) + '|' +
        best['scraped_date'].astype(str) + '|' +
        best['scraped_time']
    )

    # save cleaned data to parquet file
    os.makedirs(os.path.dirname(INTERIM), exist_ok=True)
    best.to_parquet(INTERIM, index=False)
    print(f"✓ wrote {len(best):,} rows with rich daily+time stats → {INTERIM}")

if __name__ == "__main__":
    main()



