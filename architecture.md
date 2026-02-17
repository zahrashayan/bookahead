# BookAhead System Architecture

## Overview
ML-powered flight price prediction with clear separation of concerns.

---

## Component Diagram
```
┌──────────────────────────────────────────────────┐
│         USER INTERFACE (app.py)                  │
│         • Route/date selection                   │
│         • Price visualization                    │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│      PREDICTION SERVICE (src/prediction/)        │
│      • predictor.py - Load models, predict       │
│      • recommendation.py - Booking advice        │
│      • api.py - REST API endpoints               │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│       MODEL TRAINING (src/models/)               │
│       • train_xgboost.py - XGBoost training      │
│       • train_baseline.py - Linear Regression    │
│       • train_random_forest.py - RF baseline     │
│       • evaluate.py - MAE, Dir. Accuracy         │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│     DATA PROCESSING (src/data_processing/)       │
│     • prepare_panel.py - Clean & aggregate       │
│     • feature_engineering.py - Proxy features    │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│    DATA COLLECTION (src/data_collection/)        │
│    • scraper.py - Web scraping                   │
│    • scraper_config.py - Routes & settings       │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│    CONFIGURATION (config/settings.py)            │
│    • Paths, hyperparameters, thresholds          │
└──────────────────────────────────────────────────┘
```

---

## Separation of Concerns

| Layer | Responsibility | Why Separate? |
|-------|---------------|---------------|
| **Data Collection** | Web scraping | Can change data sources without touching ML |
| **Data Processing** | Feature engineering | Can add features without retraining |
| **Model Training** | Train models offline | Experimentation doesn't affect production |
| **Prediction** | Serve predictions online | Lightweight, fast, no training overhead |
| **UI** | User interaction | Frontend changes don't affect backend |
| **Config** | Central settings | Single source of truth |

---

## Proxy Features for Airline Pricing

Added 10 proxy features to capture pricing dynamics airlines use:

**Inventory Signals:**
- `price_slope` - Rate of price change (steeper = fewer seats)
- `price_volatility_7d` - Price volatility (high = scarcity)
- `daily_price_points` - Number of distinct prices

**Competition Signals:**
- `airline_count` - Number of airlines on route
- `flight_count` - Number of flights available
- `price_cv` - Price dispersion

**Demand Signals:**
- `price_pct_change` - Price momentum
- `price_vs_min` - Distance from minimum price
- `booking_urgency` - Within 14 days flag
- `days_until_squared` - Non-linear time effect

---

## Evaluation Metrics

**Primary Metrics:**
- **MAE** - Average prediction error in dollars
- **Median AE** - Robust to outliers  
- **Directional Accuracy** - Predicting movement direction

**Asymmetric Loss:**
- Over-prediction (pred > actual): User waits, might miss deal
- Under-prediction (pred < actual): **Risky** - user books, price rises

---

## Data Flow

**Training:**
```
Raw CSV → Clean → Add Proxy Features → Train XGBoost → Evaluate (MAE) → Save Models
```

**Prediction:**
```
User Input → Load Model → Predict → Recommendation → Display
```

---

## File Structure
```
bookahead/
├── config/settings.py
├── src/
│   ├── data_collection/
│   ├── data_processing/
│   ├── models/
│   ├── prediction/
│   └── utils/
├── data/{raw, interim, processed}/
├── models/*.pkl
└── app.py
```

---