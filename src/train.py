"""
Model Training and Time-Based Evaluation Pipeline
-------------------------------------------------
Trains and benchmarks:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor

Evaluates on out-of-time test dataset (2023) using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R2 (Coefficient of Determination)

Persists trained models and metrics in the models/ directory.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import prepare_and_preprocess_data

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error (MAPE) handling small/zero denominators."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)

def train_and_evaluate_all_models(
    data_path: str = "data/processed/final_dataset.csv",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Executes full training and benchmarking lifecycle.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Preprocessed Data with Time-Based Split
    X_train, y_train, X_test, y_test, encoders, feature_cols = prepare_and_preprocess_data(
        data_path=data_path,
        split_year=2023,
        models_dir=models_dir
    )

    models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }

    results = {}
    best_model_name = None
    best_rmse = float('inf')
    best_model_obj = None

    print("\n--- Model Training & Out-of-Time Validation (Test Year: 2023) ---")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predictions on Test set (2023)
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(calculate_mape(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        file_slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_save_path = os.path.join(models_dir, f"{file_slug}.pkl")
        joblib.dump(model, model_save_path)

        results[name] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape_pct': round(mape, 2),
            'r2_score': round(r2, 4),
            'model_path': model_save_path
        }

        print(f" -> {name} | MAE: INR {mae:,.2f} | RMSE: INR {rmse:,.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

    # Save Best Model
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_obj, best_model_path)
    
    # Save test dataset actual vs predictions for visualization
    pred_comparison_df = X_test.copy()
    pred_comparison_df['actual_revenue'] = y_test.values
    for name, model in models.items():
        pred_comparison_df[f'pred_{name}'] = model.predict(X_test)
    pred_comparison_df.to_csv(os.path.join(models_dir, "test_predictions_comparison.csv"), index=False)

    summary = {
        'best_model': best_model_name,
        'best_model_path': best_model_path,
        'metrics': results
    }

    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[SUCCESS] Best Model Identified: {best_model_name} (Saved to {best_model_path})")
    return summary

if __name__ == "__main__":
    train_and_evaluate_all_models()
