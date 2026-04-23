"""
Linear Regression baseline model
Trains a simple linear regression model per route.
Updated metrics per professor feedback: removed R², added directional accuracy and regret.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
from src.common.features import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.common.splits import grouped_time_split

# path to the cleaned dataset + where to save the trained models
DATA_PATH  = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def main():
    print("Loading data from:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # numeric and categorical features + target
    num_feats = BASELINE_NUMERIC_FEATURES
    cat_feats = BASELINE_CATEGORICAL_FEATURES
    target = TARGET_COLUMN

    # make sure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # loop through each route and train a separate model
    for route in df['route'].unique():
        sub = df[df['route'] == route].copy()
        if len(sub) < 10:
            print(f"Skipping {route} (only {len(sub)} rows)")
            continue

        print(f"{'='*70}")
        print(f"{route} - Linear Regression Baseline")
        print(f"{'='*70}")
        
        # train/test split
        train, test = grouped_time_split(sub)
        X_train, y_train = train[num_feats + cat_feats], train[target]
        X_test,  y_test  = test[num_feats + cat_feats],  test[target]

        print(f"Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

        # preprocessing: scaling for numeric and one-hot for categorical
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # simple linear regression as baseline
        model = LinearRegression()

        # build pipeline
        pipe = Pipeline([
            ('pre', pre),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # ===== METRICS (Updated per professor feedback) =====
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n{'─'*70}")
        print("PERFORMANCE METRICS")
        print(f"{'─'*70}")
        print(f"MAE:  ${mae:>8.2f}  (average prediction error)")
        print(f"RMSE: ${rmse:>8.2f}  (penalizes large errors more)")

        # Directional accuracy - did we predict up/down correctly?
        if len(y_test) > 1:
            actual_changes = np.diff(y_test.values)
            pred_changes = np.diff(y_pred)
            
            # Only calculate if we have price changes
            if len(actual_changes) > 0:
                correct_direction = np.sign(actual_changes) == np.sign(pred_changes)
                dir_acc = np.mean(correct_direction) * 100
                print(f"Directional Accuracy: {dir_acc:>5.1f}%  (correct price direction)")

        # Regret: cost of following model vs optimal decision
        if len(y_test) > 1:
            predicted_best_idx = np.argmin(y_pred)
            actual_best_idx = np.argmin(y_test.values)
            
            predicted_best_price = y_test.values[predicted_best_idx]
            actual_best_price = y_test.values[actual_best_idx]
            
            regret = predicted_best_price - actual_best_price
            print(f"Regret: ${regret:>8.2f}  (extra cost if following model)")
        
        # Feature importance (top 3 coefficients)
        print(f"\n{'─'*70}")
        print("TOP 3 FEATURES BY IMPACT")
        print(f"{'─'*70}")
        
        feature_names = list(pipe.named_steps['pre'].get_feature_names_out())
        coefficients = pipe.named_steps['model'].coef_
        
        top_idx = np.argsort(np.abs(coefficients))[-3:][::-1]
        for idx in top_idx:
            if idx < len(feature_names):
                feat = feature_names[idx]
                coef = coefficients[idx]
                impact = "increases" if coef > 0 else "decreases"
                print(f"  {feat:30s}  ${coef:>8.2f}  ({impact} price)")
        
        # save model
        model_path = os.path.join(MODELS_DIR, f"lr_{route.replace('-','_')}.pkl")
        joblib.dump(pipe, model_path)
        print(f"\n✓ Model saved → {model_path}")
        print(f"{'='*70}\n")

    print("All route models trained.\n")

if __name__ == "__main__":
    main()
