# src/data_processing/prepare_panel.py
# builds a clean dataset (panel) with proxy features for airline pricing dynamics

import os
import pandas as pd
import numpy as np

# paths
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

    print(f"Loaded {len(df):,} rows")

    # ========================================================================
    # STEP 1: Calculate COMPETITION proxies (before aggregation)
    # ========================================================================
    print("Calculating competition proxies...")
    
    # Number of distinct airlines per route/scrape date (competition signal)
    airline_counts = df.groupby(['route', 'scraped_date'])['airline'].nunique().reset_index()
    airline_counts.rename(columns={'airline': 'airline_count'}, inplace=True)
    
    # Number of distinct flights per route/scrape date
    flight_counts = df.groupby(['route', 'scraped_date']).size().reset_index(name='flight_count')
    
    # Price dispersion (competition increases price variation)
    price_dispersion = df.groupby(['route', 'scraped_date'])['price'].agg(['std', 'mean']).reset_index()
    price_dispersion['price_cv'] = price_dispersion['std'] / price_dispersion['mean']
    price_dispersion['price_cv'] = price_dispersion['price_cv'].fillna(0)
    price_dispersion = price_dispersion[['route', 'scraped_date', 'price_cv']]

    # ========================================================================
    # STEP 2: Aggregate to minimum price per snapshot
    # ========================================================================
    print("Aggregating to minimum prices...")
    best = (
        df.groupby(['route','departure_date','scraped_date','scraped_time'], as_index=False)
          .agg(min_price=('price','min'))
    )

    # Merge competition features
    best = best.merge(airline_counts, on=['route', 'scraped_date'], how='left')
    best = best.merge(flight_counts, on=['route', 'scraped_date'], how='left')
    best = best.merge(price_dispersion, on=['route', 'scraped_date'], how='left')
    
    # Fill missing values
    best['airline_count'] = best['airline_count'].fillna(1)
    best['flight_count'] = best['flight_count'].fillna(1)
    best['price_cv'] = best['price_cv'].fillna(0)

    # ========================================================================
    # STEP 3: Add basic temporal features
    # ========================================================================
    print("Adding temporal features...")
    best['days_until'] = (best['departure_date'] - best['scraped_date']).dt.days
    best = best[best['days_until'] >= 0]
    best['book_dow']   = best['scraped_date'].dt.dayofweek
    best['depart_dow'] = best['departure_date'].dt.dayofweek
    best['depart_woy'] = best['departure_date'].dt.isocalendar().week.astype(int)
    best[['route_O','route_D']] = best['route'].str.split('-', expand=True)
    
    # Urgency indicator (< 14 days = high urgency)
    best['booking_urgency'] = (best['days_until'] < 14).astype(int)
    
    # Non-linear time effect
    best['days_until_squared'] = best['days_until'] ** 2

    # After the temporal features section (around line 80)

# ========================================================================
# STEP 6: Add holiday demand signals
# ========================================================================
    print("Adding holiday features...")

# Define major travel holidays
    MAJOR_HOLIDAYS = {
    'thanksgiving': ['11-24', '11-25', '11-26', '11-27'],
    'christmas': ['12-23', '12-24', '12-25', '12-26', '12-27'],
    'new_years': ['12-30', '12-31', '01-01', '01-02'],
    'spring_break': ['03-15', '03-22'],  # Typical spring break week
    'july_4th': ['07-03', '07-04', '07-05'],
    'labor_day': ['09-01', '09-02', '09-03'],
    'memorial_day': ['05-25', '05-26', '05-27']
}

    def get_holiday_category(date):
        month_day = date.strftime('%m-%d')
        for holiday_name, dates in MAJOR_HOLIDAYS.items():
            if month_day in dates:
                return holiday_name
        return 'none'

    best['holiday_category'] = best['departure_date'].apply(get_holiday_category)
    best['is_major_holiday'] = (best['holiday_category'] != 'none').astype(int)

    # ========================================================================
    # STEP 4: Add SCARCITY proxies (time-series features)
    # ========================================================================
    print("Calculating scarcity proxies (price dynamics)...")
    
    # Sort for time-series calculations
    best = best.sort_values(['route', 'departure_date', 'scraped_date'])
    
    # Price slope (Δprice/Δday) - steeper slope = fewer cheap seats
    best['price_diff'] = best.groupby(['route', 'departure_date'])['min_price'].diff()
    best['days_diff'] = best.groupby(['route', 'departure_date'])['days_until'].diff()
    best['price_slope'] = best['price_diff'] / best['days_diff'].abs()
    best['price_slope'] = best['price_slope'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Rolling volatility (7-day window) - volatility indicates scarcity
    best['price_volatility_7d'] = (
        best.groupby(['route', 'departure_date'])['min_price']
        .transform(lambda x: x.rolling(window=7, min_periods=1).std())
    )
    best['price_volatility_7d'] = best['price_volatility_7d'].fillna(0)
    
    # Price momentum (% change)
    best['price_pct_change'] = (
        best.groupby(['route', 'departure_date'])['min_price']
        .pct_change()
    )
    best['price_pct_change'] = best['price_pct_change'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Distance from minimum price for this route/departure
    best['route_min_price'] = (
        best.groupby(['route', 'departure_date'])['min_price']
        .transform('min')
    )
    best['price_vs_min'] = best['min_price'] - best['route_min_price']
    
    # Number of distinct price points per day (fewer = inventory pressure)
    daily_price_points = (
        best.groupby(['route', 'scraped_date'])['min_price']
        .nunique()
        .reset_index(name='daily_price_points')
    )
    best = best.merge(daily_price_points, on=['route', 'scraped_date'], how='left')
    best['daily_price_points'] = best['daily_price_points'].fillna(1)

    # ========================================================================
    # STEP 5: Clean up and save
    # ========================================================================
    # Drop intermediate columns
    best = best.drop(columns=['price_diff', 'days_diff', 'route_min_price'], errors='ignore')
    
    # unique id for each record
    best['panel'] = (
        best['route'] + '|' +
        best['departure_date'].astype(str) + '|' +
        best['scraped_date'].astype(str) + '|' +
        best['scraped_time']
    )

    # save cleaned data to parquet file
    os.makedirs(os.path.dirname(INTERIM), exist_ok=True)
    best.to_parquet(INTERIM, index=False)
    
    print(f"\n✓ Saved {len(best):,} rows with {len(best.columns)} features → {INTERIM}")
    print("\nNew proxy features added:")
    print("  • airline_count (competition)")
    print("  • flight_count (competition)")
    print("  • price_cv (price dispersion)")
    print("  • price_slope (inventory signal)")
    print("  • price_volatility_7d (scarcity)")
    print("  • price_pct_change (momentum)")
    print("  • price_vs_min (relative pricing)")
    print("  • daily_price_points (availability)")
    print("  • booking_urgency (< 14 days)")
    print("  • days_until_squared (non-linear time)")
    print(f"\nTotal features: {len(best.columns)}")

if __name__ == "__main__":
    main()



