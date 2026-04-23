# BookAhead

BookAhead is a flight-price timing project focused on answering a specific question:

`When is the best time to book a flight?`

The repo currently contains an end-to-end prototype pipeline that:

- collects fare snapshots for a small set of routes
- cleans and aggregates those snapshots into a modeling dataset
- engineers time-based and pricing-dynamics features
- trains route-specific baseline and XGBoost models
- saves trained artifacts and experiment outputs
- exposes a Streamlit app and FastAPI endpoint backed by a shared prediction service

This is best described as a research / product prototype rather than a production system.

## Current State

What is working today:

- Selenium-based flight scraping for a limited, hardcoded set of routes and dates
- data preparation into a parquet panel dataset
- baseline model training with Linear Regression and Random Forest
- stronger route-specific XGBoost training with engineered proxy features
- evaluation utilities for MAE, RMSE, directional accuracy, and asymmetric error analysis
- experiment scripts for hyperparameter tuning and sensitivity analysis
- a shared prediction layer that uses historical route context at inference time
- a Streamlit UI and FastAPI endpoint that both use the same prediction path

## Repository Layout

```text
bookahead/
├── app.py
├── architecture.md
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── figures/
├── models/
├── scripts/
├── src/
│   ├── common/
│   │   ├── features.py
│   │   └── splits.py
│   ├── data_collection/
│   │   ├── scraper.py
│   │   └── scraper_config.py
│   ├── data_processing/
│   │   └── prepare_panel.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── sensitivity_analysis.py
│   │   ├── train_baseline.py
│   │   ├── train_random_forest.py
│   │   └── train_xgboost.py
│   ├── prediction/
│   │   ├── api.py
│   │   ├── feature_builder.py
│   │   ├── history_repository.py
│   │   ├── model_registry.py
│   │   ├── predictor.py
│   │   ├── proxy_feature_service.py
│   │   └── schemas.py
│   └── utils/
└── README.md
```

## Data Flow

The current workflow looks like this:

```text
Selenium scraper
    -> raw CSV files
    -> cleaned / prepared panel data
    -> historical route context with engineered proxy features
    -> per-route model training
    -> saved model artifacts
    -> shared prediction service
    -> Streamlit UI and FastAPI inference
```

## Routes Covered Right Now

The current scraper config is intentionally small and route-specific. It tracks these routes:

- `SFO -> NYC`
- `SFO -> SAN`
- `SFO -> ISB`

Each route also uses a small set of hardcoded departure dates defined in `src/data_collection/scraper_config.py`.

## Modeling Approach

### Baselines

The repo includes two baseline modeling approaches:

- Linear Regression: `src/models/train_baseline.py`
- Random Forest: `src/models/train_random_forest.py`

These provide simple reference points before using the stronger XGBoost model.

### Main Model

The main modeling script is:

- `src/models/train_xgboost.py`

This script trains a separate XGBoost model per route using:

- temporal features like `days_until`, `book_dow`, `depart_dow`, `depart_woy`
- competition proxies like `airline_count`, `flight_count`, `price_cv`
- scarcity / momentum proxies like `price_slope`, `price_volatility_7d`, `price_pct_change`
- urgency and non-linear features like `booking_urgency` and `days_until_squared`
- holiday indicators such as `holiday_category` and `is_major_holiday`

### Evaluation

Evaluation logic lives in:

- `src/models/evaluate.py`

Current metrics include:

- MAE
- Median Absolute Error
- RMSE
- R2
- MAPE
- Directional Accuracy based on sequential price movement
- asymmetric over- vs under-prediction analysis

## Experiment Scripts

The repo also includes experiment / analysis scripts:

- `src/models/hyperparameter_tuning.py`
  Tests XGBoost hyperparameter combinations with time-series cross-validation.

- `src/models/sensitivity_analysis.py`
  Measures how model performance changes when proxy features are added or removed.

## App and API

The app and API now share the same core prediction path:

- request schema: `src/prediction/schemas.py`
- historical route lookup: `src/prediction/history_repository.py`
- proxy feature extraction: `src/prediction/proxy_feature_service.py`
- inference-time feature assembly: `src/prediction/feature_builder.py`
- model loading: `src/prediction/model_registry.py`
- prediction orchestration: `src/prediction/predictor.py`
- booking recommendation logic: `src/prediction/recommendation.py`

### Streamlit Demo App

The demo UI is:

- `app.py`

It currently lets you:

- choose a route
- choose a departure date
- generate a price prediction using historical route context
- view simple booking guidance
- inspect a trend-style visualization

### FastAPI Endpoint

The API prototype is:

- `src/prediction/api.py`

It exposes the shared predictor through a `/predict` endpoint using the canonical `PredictionRequest` schema.
The response includes the predicted price, historical feature context, and the same booking recommendation guidance used by the app.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

The current `requirements.txt` covers the core model / API stack.

```bash
pip install -r requirements.txt
```

Depending on what you want to run, you may also need additional packages used elsewhere in the repo:

```bash
pip install streamlit plotly selenium pyarrow matplotlib
```

If you plan to run scraping locally, you will also need a working Chrome / ChromeDriver setup that matches your Selenium environment.

## How To Run

### 1. Run the scraper

```bash
python src/data_collection/scraper.py
```

Notes:

- the scraper is currently browser-driven and route-config based
- it writes raw flight data under `data/raw/`
- it is not yet designed as a headless, production-grade ingestion service

### 2. Prepare the panel dataset

```bash
python src/data_processing/prepare_panel.py
```

This script reads the cleaned raw flight file and writes:

- `data/interim/best_today.parquet`

### 3. Train the baseline linear model

### Optional: Build the inference route-context table

```bash
python src/data_processing/build_route_context.py
```

This writes a narrower inference table to:

- `data/processed/route_context_features.parquet`

When this file exists, the prediction layer uses it for faster historical context lookups. If it does not exist, prediction falls back to `data/interim/best_today.parquet`.

### 3. Train the baseline linear model

```bash
python src/models/train_baseline.py
```

### 4. Train the Random Forest baseline

```bash
python src/models/train_random_forest.py
```

### 5. Train the XGBoost models

```bash
python src/models/train_xgboost.py
```

Saved model artifacts are written under:

- `models/`

### 6. Run hyperparameter tuning

```bash
python src/models/hyperparameter_tuning.py
```

### 7. Run sensitivity analysis

```bash
python src/models/sensitivity_analysis.py
```

### 8. Launch the Streamlit app

```bash
streamlit run app.py
```

### 9. Launch the FastAPI server

```bash
uvicorn src.prediction.api:app --reload
```

## Shared Prediction Layer

The prediction system now uses historical route context at inference time instead of relying only on calendar features.

The current flow is:

```text
User request
    -> PredictionRequest
    -> route/departure history lookup from best_today.parquet
    -> proxy feature extraction
    -> feature row assembly
    -> route-specific model load
    -> prediction result
```

This keeps the app and API aligned with the richer XGBoost feature contract used during training.

## Outputs

This repo currently stores several kinds of outputs:

- trained model artifacts in `models/`
- experiment summaries such as:
  - `models/xgboost_results_with_proxy_features.csv`
  - `models/hyperparameter_tuning_results.csv`
  - `models/sensitivity_analysis_results.csv`
- generated figures and exploratory plots

## Recommended Entry Points

If you are new to the repo, these are the best files to start with:

- `src/data_collection/scraper.py`
- `src/data_processing/prepare_panel.py`
- `src/models/train_xgboost.py`
- `src/models/evaluate.py`
- `app.py`
