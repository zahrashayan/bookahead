"""
XGBoost Model - Advanced gradient boosting for flight price prediction
Main model for the capstone project
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# load cleaned data + where to save models
DATA_PATH  = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def grouped_time_split(df, panel_col='panel', time_col='scraped_date', train_frac=0.8):
    """Split data chronologically to avoid data leakage"""
    panel_first = df.groupby(panel_col)[time_col].min().sort_values()
    panels = panel_first.index.to_list()
    cut = int(len(panels) * train_frac)
    train_panels = set(panels[:cut])
    train = df[df[panel_col].isin(train_panels)].copy()
    test  = df[~df[panel_col].isin(train_panels)].copy()
    return train, test

def main():
    print("\n" + "="*70)
    print("XGBOOST MODEL TRAINING")
    print("="*70)
    
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # feature columns
    num_feats = ['days_until','book_dow','depart_dow','depart_woy']
    cat_feats = ['route_O','route_D']
    target = 'min_price'

    os.makedirs(MODELS_DIR, exist_ok=True)

    # track results for comparison
    results = []

    # train a separate model for each route
    for route in df['route'].unique():
        sub = df[df['route'] == route].copy()
        if len(sub) < 10:
            print(f"⚠️  Skipping {route} (only {len(sub)} rows)\n")
            continue

        print(f"{'='*70}")
        print(f"ROUTE: {route}")
        print(f"{'='*70}")
        
        train, test = grouped_time_split(sub)
        print(f"Train: {len(train)} rows | Test: {len(test)} rows")
        
        X_train, y_train = train[num_feats + cat_feats], train[target]
        X_test,  y_test  = test[num_feats + cat_feats],  test[target]

        # preprocessing: one-hot encode categorical features
        pre = ColumnTransformer([
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # XGBoost parameters (tuned for small datasets)
        xgb_params = {
            'n_estimators': 100,           # number of boosting rounds
            'learning_rate': 0.05,         # step size (smaller = more careful learning)
            'max_depth': 4,                # tree depth (shallow to prevent overfitting)
            'min_child_weight': 3,         # minimum samples per leaf
            'subsample': 0.8,              # use 80% of data per tree
            'colsample_bytree': 0.8,       # use 80% of features per tree
            'gamma': 0.1,                  # regularization (penalizes complex trees)
            'reg_alpha': 0.1,              # L1 regularization
            'reg_lambda': 1.0,             # L2 regularization
            'random_state': 42,
            'verbosity': 0
        }

        model = xgb.XGBRegressor(**xgb_params)

        # pipeline
        pipe = Pipeline([
            ('pre', pre),
            ('model', model)
        ])

        print("\nTraining XGBoost...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # evaluate
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"\n{'Performance Metrics':^70}")
        print("-"*70)
        print(f"  MAE:  ${mae:>10,.2f}  (average prediction error)")
        print(f"  RMSE: ${rmse:>10,.2f}  (root mean squared error)")
        print(f"  R²:   {r2:>11.3f}  (variance explained: {r2*100:.1f}%)")
        print("-"*70)

        # feature importance (what XGBoost learned matters most)
        feature_names = num_feats + list(pipe.named_steps['pre'].transformers_[1][1].get_feature_names_out())
        importance = pipe.named_steps['model'].feature_importances_
        
        print(f"\n{'Top 5 Important Features':^70}")
        print("-"*70)
        top_idx = np.argsort(importance)[-5:][::-1]
        for i, idx in enumerate(top_idx, 1):
            if idx < len(feature_names):
                print(f"  {i}. {feature_names[idx]:<30} {importance[idx]:>8.4f}")
        print("-"*70)

        # save model
        model_path = os.path.join(MODELS_DIR, f"xgb_{route.replace('-','_')}.pkl")
        joblib.dump(pipe, model_path)
        print(f"\n✓ Saved model → {model_path}\n")

        # store results for comparison
        results.append({
            'route': route,
            'model': 'XGBoost',
            'train_size': len(train),
            'test_size': len(test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })

    # summary comparison
    print("\n" + "="*70)
    print("SUMMARY: All Routes")
    print("="*70)
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        print(f"\nAverage R²: {results_df['r2'].mean():.3f}")
        print(f"Average MAE: ${results_df['mae'].mean():.2f}")
    
    print("\n" + "="*70)
    print("✅ XGBoost training complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()