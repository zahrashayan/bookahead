# src/ml/pipeline.py
# trains a separate Random Forest model for each route using the prepared data

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# load cleaned parquet + where to save trained models
DATA_PATH  = os.path.join(os.path.dirname(__file__), '../../data/interim/best_today.parquet')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def grouped_time_split(df, panel_col='panel', time_col='scraped_date', train_frac=0.8):
    # split data chronologically (not randomly) to avoid data leakage
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
    print(f"Loaded {len(df):,} rows")

    # numerical and categorical feature columns
    num_feats = ['days_until','book_dow','depart_dow','depart_woy']
    cat_feats = ['route_O','route_D']
    target = 'min_price'

    os.makedirs(MODELS_DIR, exist_ok=True)

    # train a separate model for each route
    for route in df['route'].unique():
        sub = df[df['route'] == route].copy()
        if len(sub) < 10:
            print(f"Skipping {route} (only {len(sub)} rows)")
            continue

        print(f"\n==================== {route} ====================")
        train, test = grouped_time_split(sub)
        X_train, y_train = train[num_feats + cat_feats], train[target]
        X_test,  y_test  = test[num_feats + cat_feats],  test[target]

        # handle categorical vars with one-hot encoding
        pre = ColumnTransformer([
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # random forest = good baseline model for tabular data
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )

        # combine preprocessing + model in one pipeline
        pipe = Pipeline([
            ('pre', pre),
            ('model', model)
        ])

        print(f"Training model for {route} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # evaluate with standard regression metrics
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.3f}")

        # save the trained model so we can reuse it later
        model_path = os.path.join(MODELS_DIR, f"rf_{route.replace('-','_')}.pkl")
        joblib.dump(pipe, model_path)
        print(f"✓ Saved model for {route} → {model_path}")

    print("\n All available route models trained and saved.\n")

if __name__ == "__main__":
    main()



