"""
XGBoost Main Model - Advanced gradient boosting for flight price prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# Import our new evaluation module
from src.models.evaluate import evaluate_model, print_evaluation_report

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
    print("XGBOOST MODEL TRAINING - WITH PROXY FEATURES")
    print("="*70)
    
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} features\n")

    # ========================================================================
    # FEATURE SELECTION (INCLUDING PROXY FEATURES)
    # ========================================================================
    
    # Basic temporal features
    base_features = ['days_until', 'book_dow', 'depart_dow', 'depart_woy']
    
    # Proxy features for airline pricing dynamics
    proxy_features = [
        'price_slope',           # Inventory signal (steeper = fewer seats)
        'price_volatility_7d',   # Scarcity signal
        'price_pct_change',      # Momentum
        'price_vs_min',          # Relative pricing
        'daily_price_points',    # Availability
        'airline_count',         # Competition
        'flight_count',          # Competition
        'price_cv',              # Price dispersion
        'booking_urgency',       # < 14 days flag
        'days_until_squared'     # Non-linear time effect
    ]
    
    # Combine all numerical features
    num_feats = base_features + proxy_features + ['is_major_holiday']

    # Categorical features
    cat_feats = ['route_O', 'route_D', 'holiday_category']
    
    target = 'min_price'

    print("Feature breakdown:")
    print(f"  • Base features: {len(base_features)}")
    print(f"  • Proxy features: {len(proxy_features)}")
    print(f"  • Categorical: {len(cat_feats)}")
    print(f"  • Total: {len(num_feats) + len(cat_feats)}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Track results for comparison
    results = []

    # ========================================================================
    # TRAIN MODEL FOR EACH ROUTE
    # ========================================================================
    for route in sorted(df['route'].unique()):
        sub = df[df['route'] == route].copy()
        if len(sub) < 10:
            print(f"Skipping {route} (only {len(sub)} rows)\n")
            continue

        print(f"{'='*70}")
        print(f"ROUTE: {route}")
        print(f"{'='*70}")
        
        train, test = grouped_time_split(sub)
        print(f"Train: {len(train)} rows | Test: {len(test)} rows")
        
        X_train, y_train = train[num_feats + cat_feats], train[target]
        X_test,  y_test  = test[num_feats + cat_feats],  test[target]

        # Preprocessing: one-hot encode categorical features
        pre = ColumnTransformer([
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # XGBoost parameters
        xgb_params = {
            'n_estimators': 150,           # Increased for more features
            'learning_rate': 0.05,
            'max_depth': 5,                # Slightly deeper for more features
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }

        model = xgb.XGBRegressor(**xgb_params)

        # Pipeline
        pipe = Pipeline([
            ('pre', pre),
            ('model', model)
        ])

        print("\nTraining XGBoost with proxy features...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # ====================================================================
        # EVALUATION OF MODEL
        # ====================================================================
        eval_results = evaluate_model(y_test, y_pred, model_name=f"XGBoost-{route}")
        
        # Print detailed report
        print_evaluation_report(eval_results)

        # Feature importance
        feature_names = num_feats + list(
            pipe.named_steps['pre'].transformers_[1][1].get_feature_names_out()
        )
        importance = pipe.named_steps['model'].feature_importances_
        
        print(f"\n{'Top 10 Important Features':^70}")
        print("-"*70)
        top_idx = np.argsort(importance)[-10:][::-1]
        for i, idx in enumerate(top_idx, 1):
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                # Mark proxy features with *
                marker = " *" if feat_name in proxy_features else ""
                print(f"  {i:2d}. {feat_name:<35} {importance[idx]:>8.4f}{marker}")
        print("-"*70)
        print("  * = Proxy feature for pricing dynamics")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"xgb_{route.replace('-','_')}.pkl")
        joblib.dump(pipe, model_path)
        print(f"\n✓ Saved model → {model_path}\n")

        # Store results
        results.append({
            'Route': route,
            'Train_Size': len(train),
            'Test_Size': len(test),
            'MAE': eval_results['MAE'],
            'Median_AE': eval_results['Median_AE'],
            'Dir_Acc': eval_results['Directional_Accuracy'],
            'R2': eval_results['R2'],
            'Over_Pred': eval_results['Avg_Over_Prediction'],
            'Under_Pred': eval_results['Avg_Under_Prediction']
        })

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY: All Routes")
    print("="*70)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Format for display
        print("\nPerformance Summary:")
        print("-"*70)
        for _, row in results_df.iterrows():
            print(f"{row['Route']:10s} | MAE: ${row['MAE']:6.2f} | "
          f"Median AE: ${row['Median_AE']:6.2f} | Dir.Acc: {row['Dir_Acc']:.1%}")
        print("-"*70)
        
        print(f"\nAggregate Metrics:")
        print(f"  Average MAE:           ${results_df['MAE'].mean():.2f}")
        print(f"  Average Dir. Accuracy:  {results_df['Dir_Acc'].mean():.1%}")
        
        # Save results to CSV 
        results_path = os.path.join(MODELS_DIR, 'xgboost_results_with_proxy_features.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved → {results_path}")
    
    print("\n" + "="*70)
    print("XGBoost training complete with proxy features!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()