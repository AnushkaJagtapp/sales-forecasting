"""
Data Preprocessing and Time-Based Splitting Module
-------------------------------------------------
Handles data ingestion, categorical label encoding, feature assembly,
and strict time-based train/test splitting to prevent future-data leakage.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from src.features import create_features

CATEGORICAL_COLS = ['region', 'category', 'channel', 'disaster_type']

FEATURE_COLUMNS = [
    # Categoricals (encoded)
    'region_encoded', 'category_encoded', 'channel_encoded', 'disaster_type_encoded',
    
    # Calendar & Cyclical
    'month', 'quarter', 'week_of_year', 'week_sin', 'week_cos', 'month_sin', 'month_cos',
    'is_festival_season', 'is_harvest_season', 'is_offline', 'is_online',
    
    # Pricing & Baseline Demand
    'price', 'discount_pct', 'industry_sensitivity',
    
    # Disaster Shock & Composite Impact Score
    'is_disaster_active', 'disaster_severity', 'disaster_duration_days',
    'supply_disruption_index', 'economic_disruption_index', 'geographic_spread_index',
    'population_affected_index', 'mobility_reduction_pct', 'survey_channel_shift_pct',
    'disaster_impact_score',
    
    # Interaction Features
    'disaster_x_industry_interaction', 'disaster_x_duration', 'supply_x_offline_shock',
    'mobility_x_offline_shock', 'survey_shift_x_online', 'agri_disruption_interaction',
    'ai_recession_tech_shock',
    
    # Lags and Rolling Statistics
    'lag_rev_1', 'lag_rev_2', 'lag_rev_4', 'lag_rev_8',
    'lag_units_1', 'lag_units_2', 'lag_units_4', 'lag_units_8',
    'rolling_mean_rev_4', 'rolling_std_rev_4', 'rolling_mean_rev_8', 'rolling_mean_rev_12',
    'rolling_mean_units_4'
]

TARGET_COL = 'revenue_inr'

def prepare_and_preprocess_data(
    data_path: str = "data/processed/final_dataset.csv",
    split_year: int = 2023,
    models_dir: str = "models"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    """
    Loads dataset, applies feature engineering, encodes categoricals, and splits data by time.

    Parameters:
    -----------
    data_path: path to CSV dataset
    split_year: data before split_year is Train; split_year and later is Test.
    models_dir: directory to persist fitted LabelEncoders.

    Returns:
    --------
    (X_train, y_train, X_test, y_test, encoders, feature_columns)
    """
    os.makedirs(models_dir, exist_ok=True)
    df_raw = pd.read_csv(data_path)
    
    # Apply full feature engineering
    df_featured = create_features(df_raw)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_featured[f"{col}_encoded"] = le.fit_transform(df_featured[col].astype(str))
        encoders[col] = le
        joblib.dump(le, os.path.join(models_dir, f"{col}_encoder.pkl"))

    # Time-based Train / Test Split
    train_df = df_featured[df_featured['year'] < split_year].copy()
    test_df = df_featured[df_featured['year'] >= split_year].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    # Save feature columns
    joblib.dump(FEATURE_COLUMNS, os.path.join(models_dir, "feature_columns.pkl"))
    
    # Save the processed featured dataset for app quick inspection
    df_featured.to_csv("data/processed/featured_dataset.csv", index=False)

    print(f"Dataset Preprocessing Complete!")
    print(f"Total Records: {len(df_featured)} | Train ({min(train_df['year'])}-{split_year-1}): {len(X_train)} | Test ({split_year}+): {len(X_test)}")
    print(f"Number of Features: {len(FEATURE_COLUMNS)}")

    return X_train, y_train, X_test, y_test, encoders, FEATURE_COLUMNS

if __name__ == "__main__":
    prepare_and_preprocess_data()
