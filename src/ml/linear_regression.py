"""
Linear Regression baseline model
Trains a simple linear regression model per route for comparison with Random Forest.
I’m using this to see how well a basic linear model performs before jumping into more complex ones.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# path to the cleaned dataset + where to save the trained models
DATA_PATH  = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def grouped_time_split(df, panel_col='panel', time_col='scraped_date', train_frac=0.8):
    # splitting chronologically (not randomly) so the model doesn’t “peek” into the future
    panel_first = df.groupby(panel_col)[time_col].min().sort_values()
    panels = panel_first.index.to_list()
    cut = int(len(panels) * train_frac)
    train_panels = set(panels[:cut])
    train = df[df[panel_col].isin(train_panels)].copy()
    test  = df[~df[panel_col].isin(train_panels)].copy()
    return train, test

def main():
    print("Loading data from:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # numeric and categorical features + target
    num_feats = ['days_until','book_dow','depart_dow','depart_woy']
    cat_feats = ['route_O','route_D']
    target = 'min_price'

    # make sure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # loop through each route and train a separate model
    for route in df['route'].unique():
        sub = df[df['route'] == route].copy()
        if len(sub) < 10:
            # if too few rows, skip it (model wouldn’t be reliable)
            print(f"Skipping {route} (only {len(sub)} rows)")
            continue

        print(f"==================== {route} (Linear Regression) ====================")
        
        # train/test split
        train, test = grouped_time_split(sub)
        X_train, y_train = train[num_feats + cat_feats], train[target]
        X_test,  y_test  = test[num_feats + cat_feats],  test[target]

        # preprocessing: scaling for numeric and one-hot for categorical
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),  # scaling helps linear regression
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # simple linear regression as baseline
        model = LinearRegression()

        # build pipeline (clean way to tie preprocessing + model)
        pipe = Pipeline([
            ('pre', pre),
            ('model', model)
        ])

        print(f"Training Linear Regression for {route}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # calculate metrics to see how it performs
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"  MAE:  ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²:   {r2:.3f}")
        
        # check which features matter most (just to understand what it learned)
        feature_names = num_feats + list(pipe.named_steps['pre'].get_feature_names_out())
        coefficients = pipe.named_steps['model'].coef_
        
        print(f"\n  Top 3 features by coefficient magnitude:")
        top_idx = np.argsort(np.abs(coefficients))[-3:][::-1]
        for idx in top_idx:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]}: {coefficients[idx]:>8.2f}")
        
        # save model so we can reuse it later
        model_path = os.path.join(MODELS_DIR, f"lr_{route.replace('-','_')}.pkl")
        joblib.dump(pipe, model_path)
        print(f"\n  ✓ Saved model → {model_path}\n")

    print("All route models trained.\n")

if __name__ == "__main__":
    main()
