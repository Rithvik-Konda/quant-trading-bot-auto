"""
0DTE Options Engine — Range Predictor
Predicts SPX intraday high-low range as % of open.
Used to size iron condor strikes.
"""
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

CACHE_DIR = os.path.expanduser("~/ai_trading_bot_v2/cache_options")
MODEL_PATH = os.path.join(CACHE_DIR, "range_predictor.joblib")

FEATURES = [
    'vix', 'vix1d', 'skew', 'realized_vol_5d',
    'overnight_gap_pct', 'abs_gap', 'vix_term',
    'dow', 'prior_range_pct',
    'vix_lag1', 'range_lag1', 'range_lag2', 'range_lag3',
    'vix1d_ma5', 'range_ma5', 'high_vix_flag', 'high_skew_flag',
]

def build_model_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['vix_lag1']      = df['vix'].shift(1)
    df['range_lag1']    = df['day_range_pct'].shift(1)
    df['range_lag2']    = df['day_range_pct'].shift(2)
    df['range_lag3']    = df['day_range_pct'].shift(3)
    df['vix1d_ma5']     = df['vix1d'].rolling(5).mean()
    df['range_ma5']     = df['day_range_pct'].rolling(5).mean()
    df['high_vix_flag'] = (df['vix'] > 20).astype(int)
    df['high_skew_flag']= (df['skew'] > 130).astype(int)
    return df

def train(df=None):
    if df is None:
        df = pd.read_parquet(os.path.join(CACHE_DIR, "daily_features.parquet"))
    
    df = build_model_features(df)
    df = df.dropna(subset=['day_range_pct'] + ['vix','realized_vol_5d','overnight_gap_pct'])
    
    # Use available features (vix1d/skew may be NaN for early dates)
    avail_feats = [f for f in FEATURES if f in df.columns]
    X = df[avail_feats].fillna(df[avail_feats].median())
    y = df['day_range_pct']
    
    print(f"Training on {len(df)} days, {len(avail_feats)} features")
    print(f"Target stats: mean={y.mean():.2f}% std={y.std():.2f}% max={y.max():.2f}%")
    
    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        print(f"  Fold {fold+1}: MAE={mae:.3f}% (predicting range)")
    
    print(f"CV MAE: {np.mean(maes):.3f}% ± {np.std(maes):.3f}%")
    
    # Train final model on all data
    final_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    final_model.fit(X, y)
    
    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=avail_feats).sort_values(ascending=False)
    print("\nTop features:")
    print(imp.head(10))
    
    joblib.dump({'model': final_model, 'features': avail_feats}, MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")
    return final_model, avail_feats

def predict(features_dict):
    bundle = joblib.load(MODEL_PATH)
    model, feat_cols = bundle['model'], bundle['features']
    X = pd.DataFrame([features_dict])[feat_cols].fillna(0)
    return float(model.predict(X)[0])

if __name__ == "__main__":
    train()
