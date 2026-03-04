"""
Hyperparameter Tuning with Cross-Validation
Systematically tests different model settings to find optimal configuration
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
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from src.models.evaluate import evaluate_model

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')

def cross_validate_with_time_splits(df, params, num_feats, cat_feats, route, n_splits=5):
    """
    Cross-validation using time-series splits (respects time order)
    """
    sub = df[df['route'] == route].copy()
    sub = sub.sort_values('scraped_date')  # Sort by time
    
    X = sub[num_feats + cat_feats]
    y = sub['min_price']
    
    # Time series split (respects chronological order)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    mae_scores = []
    dir_acc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Preprocessing
        pre = ColumnTransformer([
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])
        
        # Model with these specific params
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        
        pipe = Pipeline([('pre', pre), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        results = evaluate_model(y_test, y_pred)
        mae_scores.append(results['MAE'])
        dir_acc_scores.append(results['Directional_Accuracy'])
    
    return {
        'avg_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores),
        'avg_dir_acc': np.mean(dir_acc_scores),
        'std_dir_acc': np.std(dir_acc_scores)
    }

def main():
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
    print("="*70)
    
    df = pd.read_parquet(DATA_PATH)
    
    # Features
    base_features = ['days_until', 'book_dow', 'depart_dow', 'depart_woy']
    proxy_features = [
        'price_slope', 'price_volatility_7d', 'price_pct_change',
        'price_vs_min', 'daily_price_points', 'airline_count',
        'booking_urgency', 'days_until_squared'
    ]
    num_feats = base_features + proxy_features
    cat_feats = ['route_O', 'route_D']
    
    # HYPERPARAMETER GRID - Test these combinations
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 150, 200],
        'min_child_weight': [1, 3, 5]
    }
    
    # Generate all combinations
    from itertools import product
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    print(f"Testing {len(combinations)} hyperparameter combinations")
    print(f"Using 5-fold time-series cross-validation\n")
    
    routes = sorted(df['route'].unique())
    all_results = []
    
    # Test each combination
    for i, params in enumerate(combinations, 1):
        # Add fixed params
        full_params = {
            **params,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1
        }
        
        print(f"\n[{i}/{len(combinations)}] Testing:")
        print(f"  lr={params['learning_rate']}, depth={params['max_depth']}, "
              f"trees={params['n_estimators']}, min_child={params['min_child_weight']}")
        
        route_maes = []
        route_dir_accs = []
        
        for route in routes:
            cv_results = cross_validate_with_time_splits(
                df, full_params, num_feats, cat_feats, route, n_splits=5
            )
            route_maes.append(cv_results['avg_mae'])
            route_dir_accs.append(cv_results['avg_dir_acc'])
            
            print(f"    {route}: MAE=${cv_results['avg_mae']:.2f} ± ${cv_results['std_mae']:.2f}")
        
        avg_mae = np.mean(route_maes)
        avg_dir_acc = np.mean(route_dir_accs)
        
        all_results.append({
            'learning_rate': params['learning_rate'],
            'max_depth': params['max_depth'],
            'n_estimators': params['n_estimators'],
            'min_child_weight': params['min_child_weight'],
            'avg_mae': avg_mae,
            'avg_dir_acc': avg_dir_acc
        })
        
        print(f"  → Average MAE: ${avg_mae:.2f} | Dir.Acc: {avg_dir_acc:.1%}")
    
    # RESULTS
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('avg_mae')
    
    print("\nTop 10 Best Configurations (by MAE):")
    print("-"*70)
    print(results_df.head(10).to_string(index=False))
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print("⭐ BEST CONFIGURATION:")
    print("="*70)
    print(f"  Learning Rate:     {best['learning_rate']}")
    print(f"  Max Depth:         {int(best['max_depth'])}")
    print(f"  N Estimators:      {int(best['n_estimators'])}")
    print(f"  Min Child Weight:  {int(best['min_child_weight'])}")
    print(f"\n  Cross-Val MAE:     ${best['avg_mae']:.2f}")
    print(f"  Cross-Val Dir.Acc: {best['avg_dir_acc']:.1%}")
    print("="*70)
    
    # Save results
    output_path = 'models/hyperparameter_tuning_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Full results saved → {output_path}")

if __name__ == "__main__":
    main()