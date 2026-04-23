"""
Sensitivity Analysis
Tests how MAE and Directional Accuracy change when features are added/removed
Directly addresses professor feedback
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
from src.common.features import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PROXY_NUMERIC_FEATURES,
    TARGET_COLUMN,
    XGBOOST_NUMERIC_FEATURES,
)
from src.common.splits import grouped_time_split
from src.models.evaluate import evaluate_model

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')

def train_and_evaluate(df, num_feats, cat_feats, route):
    """Train model and return MAE + Directional Accuracy"""
    sub = df[df['route'] == route].copy()
    train, test = grouped_time_split(sub)
    
    X_train = train[num_feats + cat_feats]
    y_train = train[TARGET_COLUMN]
    X_test  = test[num_feats + cat_feats]
    y_test  = test[TARGET_COLUMN]

    pre = ColumnTransformer([
        ('num', 'passthrough', num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])

    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    pipe = Pipeline([('pre', pre), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results = evaluate_model(y_test, y_pred)
    return results['MAE'], results['Directional_Accuracy']

def main():
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # All features
    base_features = BASE_NUMERIC_FEATURES
    proxy_features = PROXY_NUMERIC_FEATURES
    cat_feats = CATEGORICAL_FEATURES
    all_num_feats = XGBOOST_NUMERIC_FEATURES

    routes = sorted(df['route'].unique())

    # ================================================================
    # TEST 1: Baseline (base features only - no proxies)
    # ================================================================
    print("Running baseline (base features only)...")
    baseline_results = {}
    for route in routes:
        mae, dir_acc = train_and_evaluate(df, base_features, cat_feats, route)
        baseline_results[route] = {'MAE': mae, 'Dir_Acc': dir_acc}

    # ================================================================
    # TEST 2: Full model (all features)
    # ================================================================
    print("Running full model (all features)...")
    full_results = {}
    for route in routes:
        mae, dir_acc = train_and_evaluate(df, all_num_feats, cat_feats, route)
        full_results[route] = {'MAE': mae, 'Dir_Acc': dir_acc}

    # ================================================================
    # TEST 3: Remove one feature at a time
    # ================================================================
    print("Running leave-one-out analysis...\n")
    
    sensitivity_rows = []

    for feature in proxy_features:
        # Features WITHOUT this one
        reduced_feats = [f for f in all_num_feats if f != feature]
        
        route_maes = []
        route_dir_accs = []

        for route in routes:
            mae, dir_acc = train_and_evaluate(df, reduced_feats, cat_feats, route)
            route_maes.append(mae)
            route_dir_accs.append(dir_acc)

        avg_mae = np.mean(route_maes)
        avg_dir_acc = np.mean(route_dir_accs)

        # Compare to full model
        full_avg_mae = np.mean([full_results[r]['MAE'] for r in routes])
        full_avg_dir = np.mean([full_results[r]['Dir_Acc'] for r in routes])

        mae_impact = avg_mae - full_avg_mae       # Positive = removing hurts MAE
        dir_impact = avg_dir_acc - full_avg_dir   # Negative = removing hurts Dir.Acc

        sensitivity_rows.append({
            'Feature Removed': feature,
            'MAE Without': round(avg_mae, 2),
            'MAE Impact': round(mae_impact, 2),
            'Dir.Acc Without': round(avg_dir_acc, 3),
            'Dir.Acc Impact': round(dir_impact, 3),
            'Important?': ' YES' if mae_impact > 1.0 else 'No'
        })
        
        print(f"  Removed {feature:<25} MAE: ${avg_mae:.2f} (impact: {mae_impact:+.2f})")

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*70)

    # Baseline vs Full Model
    baseline_avg_mae = np.mean([baseline_results[r]['MAE'] for r in routes])
    baseline_avg_dir = np.mean([baseline_results[r]['Dir_Acc'] for r in routes])
    full_avg_mae = np.mean([full_results[r]['MAE'] for r in routes])
    full_avg_dir = np.mean([full_results[r]['Dir_Acc'] for r in routes])

    print("\n Baseline vs Full Model:")
    print("-"*70)
    print(f"  Baseline (4 features):    MAE: ${baseline_avg_mae:.2f} | Dir.Acc: {baseline_avg_dir:.1%}")
    print(f"  Full Model (14 features): MAE: ${full_avg_mae:.2f} | Dir.Acc: {full_avg_dir:.1%}")
    print(f"  Improvement:              MAE: ${baseline_avg_mae - full_avg_mae:.2f} better | Dir.Acc: {full_avg_dir - baseline_avg_dir:.1%} better")

    print("\n Leave-One-Out Feature Analysis (sorted by impact):")
    print("-"*70)
    
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df = sensitivity_df.sort_values('MAE Impact', ascending=False)
    
    print(sensitivity_df.to_string(index=False))

    print("\n Per-Route Summary:")
    print("-"*70)
    print(f"\n{'Route':<12} {'Baseline MAE':>14} {'Full MAE':>10} {'Improvement':>12} {'Dir.Acc':>10}")
    print("-"*70)
    for route in routes:
        b_mae = baseline_results[route]['MAE']
        f_mae = full_results[route]['MAE']
        improvement = b_mae - f_mae
        dir_acc = full_results[route]['Dir_Acc']
        print(f"{route:<12} ${b_mae:>12.2f} ${f_mae:>8.2f} ${improvement:>10.2f} {dir_acc:>9.1%}")

    # Save results
    output_path = 'models/sensitivity_analysis_results.csv'
    sensitivity_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved → {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
