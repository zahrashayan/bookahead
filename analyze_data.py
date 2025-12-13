"""
BookAhead Data Analysis Script
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load  data
df = pd.read_csv('data/raw/flights_data_cleaned.csv')

print("=" * 80)
print("BOOKAHEAD DATA ANALYSIS REPORT")
print("=" * 80)

# ============================================================================
# 1. DATASET SIZE & ROUTE DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("1. DATASET SIZE & ROUTE DISTRIBUTION")
print("=" * 80)

print(f"\nTotal rows in dataset: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names: {df.columns.tolist()}")

# Route analysis
if 'route' in df.columns:
    print(f"\nUnique routes: {df['route'].nunique()}")
    print("\nSamples per route:")
    route_counts = df['route'].value_counts()
    print(route_counts)
    print("\nRoute distribution (%):")
    print((route_counts / len(df) * 100).round(2))
elif 'route_O' in df.columns and 'route_D' in df.columns:
    df['route'] = df['route_O'] + '-' + df['route_D']
    print(f"\nUnique routes: {df['route'].nunique()}")
    print("\nSamples per route:")
    route_counts = df['route'].value_counts()
    print(route_counts)
    print("\nRoute distribution (%):")
    print((route_counts / len(df) * 100).round(2))
else:
    print("\nNo route column found!")

# ============================================================================
# 2. DATE RANGES & TEMPORAL COVERAGE
# ============================================================================
print("\n" + "=" * 80)
print("2. DATE RANGES & TEMPORAL COVERAGE")
print("=" * 80)

# Try to find date columns
date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
print(f"\nDate-related columns found: {date_cols}")

# Analyze scraping date if exists
scrape_date_col = None
for col in ['scrape_date', 'scraped_date', 'scraping_date']:
    if col in df.columns:
        scrape_date_col = col
        break

if scrape_date_col:
    df[scrape_date_col] = pd.to_datetime(df[scrape_date_col])
    print(f"\nScraping period:")
    print(f"  First scraped: {df[scrape_date_col].min()}")
    print(f"  Last scraped: {df[scrape_date_col].max()}")
    print(f"  Days of scraping: {(df[scrape_date_col].max() - df[scrape_date_col].min()).days}")
    print(f"\nUnique scraping days: {df[scrape_date_col].dt.date.nunique()}")

# Analyze departure date if exists
depart_date_col = None
for col in ['depart_date', 'departure_date', 'flight_date']:
    if col in df.columns:
        depart_date_col = col
        break

if depart_date_col:
    df[depart_date_col] = pd.to_datetime(df[depart_date_col])
    print(f"\nDeparture date range:")
    print(f"  Earliest departure: {df[depart_date_col].min()}")
    print(f"  Latest departure: {df[depart_date_col].max()}")
    print(f"  Span: {(df[depart_date_col].max() - df[depart_date_col].min()).days} days")

# Days until departure analysis
days_until_col = None
for col in ['days_until', 'days_until_departure', 'days_ahead']:
    if col in df.columns:
        days_until_col = col
        break

if days_until_col:
    print(f"\nDays until departure statistics:")
    print(df[days_until_col].describe())
    print(f"\nBooking window coverage:")
    print(f"  Min days ahead: {df[days_until_col].min()}")
    print(f"  Max days ahead: {df[days_until_col].max()}")
    print(f"  Median: {df[days_until_col].median()}")

# ============================================================================
# 3. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. MISSING DATA ANALYSIS")
print("=" * 80)

print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

if missing.sum() == 0:
    print("✓ No missing values found!")

# ============================================================================
# 4. DUPLICATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. DUPLICATE ANALYSIS")
print("=" * 80)

duplicates = df.duplicated().sum()
print(f"\nTotal duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

if duplicates > 0:
    print("\nExample duplicate rows:")
    print(df[df.duplicated(keep=False)].head())

# Check for duplicate entries (same route, date, etc.)
if scrape_date_col and depart_date_col and 'route' in df.columns:
    logical_dupes = df.duplicated(subset=['route', depart_date_col, scrape_date_col]).sum()
    print(f"\nLogical duplicates (same route/date/scrape): {logical_dupes}")

# ============================================================================
# 5. PRICE ANALYSIS & VOLATILITY
# ============================================================================
print("\n" + "=" * 80)
print("5. PRICE ANALYSIS & VOLATILITY")
print("=" * 80)

# Find price column
price_col = None
for col in ['price', 'min_price', 'Price', 'MIN_PRICE']:
    if col in df.columns:
        price_col = col
        break

if price_col:
    print(f"\nPrice statistics (using column: {price_col}):")
    print(df[price_col].describe())
    
    print(f"\nPrice range per route:")
    if 'route' in df.columns:
        route_price_stats = df.groupby('route')[price_col].agg(['min', 'max', 'mean', 'std', 'count'])
        print(route_price_stats)
        
        print(f"\nPrice volatility (Coefficient of Variation) per route:")
        cv = (route_price_stats['std'] / route_price_stats['mean'] * 100).round(2)
        print(cv.sort_values(ascending=False))
    
    # Check for price outliers
    Q1 = df[price_col].quantile(0.25)
    Q3 = df[price_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[price_col] < (Q1 - 1.5 * IQR)) | (df[price_col] > (Q3 + 1.5 * IQR))).sum()
    print(f"\nPrice outliers (using IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")
    
    # Zero or negative prices
    zero_prices = (df[price_col] <= 0).sum()
    if zero_prices > 0:
        print(f"\n⚠️ WARNING: {zero_prices} rows with zero or negative prices!")
else:
    print("\nNo price column found!")

# ============================================================================
# 6. DATA QUALITY ISSUES
# ============================================================================
print("\n" + "=" * 80)
print("6. DATA QUALITY ISSUES")
print("=" * 80)

issues = []

# Check route imbalance
if 'route' in df.columns:
    route_counts = df['route'].value_counts()
    min_samples = route_counts.min()
    max_samples = route_counts.max()
    imbalance_ratio = max_samples / min_samples
    
    if imbalance_ratio > 3:
        issues.append(f"⚠️ Route imbalance: {imbalance_ratio:.1f}x difference between routes")
        print(f"\n⚠️ Route representation is uneven:")
        print(f"  Most samples: {route_counts.index[0]} ({max_samples} rows)")
        print(f"  Least samples: {route_counts.index[-1]} ({min_samples} rows)")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    else:
        print("\n✓ Route representation is reasonably balanced")

# Check for inconsistent timestamps
if scrape_date_col and depart_date_col:
    invalid_dates = (df[scrape_date_col] > df[depart_date_col]).sum()
    if invalid_dates > 0:
        issues.append(f"⚠️ {invalid_dates} rows where scrape_date > depart_date (impossible)")
        print(f"\n⚠️ {invalid_dates} rows with scrape_date after depart_date!")

# ============================================================================
# 7. FEATURE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. FEATURE ANALYSIS")
print("=" * 80)

# Day of week distribution
dow_col = None
for col in ['depart_dow', 'day_of_week', 'departure_dow']:
    if col in df.columns:
        dow_col = col
        break

if dow_col:
    print(f"\nDeparture day of week distribution:")
    dow_dist = df[dow_col].value_counts()
    print(dow_dist)
    
    # Check if it's string or numeric
    if df[dow_col].dtype == 'object' or isinstance(df[dow_col].iloc[0], str):
        # String format
        print(f"\nDay of week breakdown:")
        for day, count in dow_dist.items():
            print(f"  {day}: {count} ({count/len(df)*100:.1f}%)")
    else:
        # Numeric format
        dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"\nDay of week breakdown:")
        dow_dist_sorted = dow_dist.sort_index()
        for idx, count in dow_dist_sorted.items():
            if 0 <= idx <= 6:
                print(f"  {dow_labels[idx]}: {count} ({count/len(df)*100:.1f}%)")

# Weekend analysis
if 'is_weekend' in df.columns:
    print(f"\nWeekend vs Weekday:")
    weekend_dist = df['is_weekend'].value_counts()
    print(weekend_dist)
    print(f"  Weekend flights: {weekend_dist.get(True, 0)} ({weekend_dist.get(True, 0)/len(df)*100:.1f}%)")
    print(f"  Weekday flights: {weekend_dist.get(False, 0)} ({weekend_dist.get(False, 0)/len(df)*100:.1f}%)")

# Stops analysis if exists
if 'stops' in df.columns:
    print(f"\nStops distribution:")
    stops_dist = df['stops'].value_counts().sort_index()
    print(stops_dist)
    if df['stops'].isnull().any():
        print(f"  ⚠️ {df['stops'].isnull().sum()} rows with missing stops data")

# Airline analysis if exists
if 'airline' in df.columns:
    print(f"\nTop 10 airlines:")
    print(df['airline'].value_counts().head(10))
    print(f"\nTotal unique airlines/combinations: {df['airline'].nunique()}")

# ============================================================================
# 8. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("8. SUMMARY & RECOMMENDATIONS")
print("=" * 80)

if issues:
    print("\n⚠️ ISSUES DETECTED:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ No major data quality issues detected!")

print("\nDataset characteristics:")
print(f"  • Total samples: {len(df)}")
if 'route' in df.columns:
    print(f"  • Routes covered: {df['route'].nunique()}")
    print(f"  • Average samples per route: {len(df) / df['route'].nunique():.0f}")
if price_col:
    print(f"  • Price range: ${df[price_col].min():.2f} - ${df[price_col].max():.2f}")
if days_until_col:
    print(f"  • Booking window: {df[days_until_col].min()}-{df[days_until_col].max()} days ahead")
if scrape_date_col:
    print(f"  • Scraping period: {(df[scrape_date_col].max() - df[scrape_date_col].min()).days} days")

print("\nKey findings:")
print(f"  • Missing values: {missing.sum()} total ({missing.sum()/len(df)*100:.2f}%)")
print(f"  • Duplicates: {duplicates} rows ({duplicates/len(df)*100:.2f}%)")
if price_col:
    print(f"  • Price outliers: {outliers} ({outliers/len(df)*100:.2f}%)")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)