# BookAhead System Architecture

## Overview
ML-powered flight price prediction with a shared, history-backed inference path.

---

## Component Diagram
```
┌──────────────────────────────────────────────────┐
│         USER INTERFACE (app.py)                  │
│         • Route/date selection                   │
│         • Price visualization                    │
│         • Uses shared predictor                  │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│      PREDICTION SERVICE (src/prediction/)        │
│      • schemas.py - Request/response contract    │
│      • predictor.py - Orchestrates inference     │
│      • model_registry.py - Load cached models    │
│      • feature_builder.py - Assemble features    │
│      • proxy_feature_service.py - Proxy signals  │
│      • history_repository.py - Route history     │
│      • api.py - REST API endpoints               │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│      SHARED CONTRACTS (src/common/)              │
│      • features.py - Canonical feature schema    │
│      • splits.py - Shared train/test split       │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│       MODEL TRAINING (src/models/)               │
│       • train_xgboost.py - XGBoost training      │
│       • train_baseline.py - Linear Regression    │
│       • train_random_forest.py - RF baseline     │
│       • hyperparameter_tuning.py - CV search     │
│       • sensitivity_analysis.py - Ablations      │
│       • evaluate.py - Shared evaluation metrics  │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│     DATA PROCESSING (src/data_processing/)       │
│     • prepare_panel.py - Clean, aggregate,       │
│       and engineer route-level features          │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│    DATA COLLECTION (src/data_collection/)        │
│    • scraper.py - Web scraping                   │
│    • scraper_config.py - Routes & settings       │
└──────────────────────────────────────────────────┘
```

---

## Separation of Concerns

| Layer | Responsibility | Why Separate? |
|-------|---------------|---------------|
| **Data Collection** | Web scraping | Can change data sources without touching ML |
| **Data Processing** | Feature engineering | Can add features without retraining |
| **Shared Contracts** | Canonical features and split logic | Keeps training and inference aligned |
| **Model Training** | Train models offline | Experimentation doesn't affect production |
| **Prediction** | Serve predictions online | Reuses route history and one shared predictor |
| **UI** | User interaction | Frontend changes don't affect backend |

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
- **Directional Accuracy** - Matching sequential movement direction

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
User Input → Request Schema → Historical Route Lookup → Proxy Features → Build Feature Row → Load Model → Predict → Display
```

---

## File Structure
```
bookahead/
├── src/
│   ├── common/
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
