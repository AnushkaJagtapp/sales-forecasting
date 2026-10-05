"""
Feature Engineering Module for Disaster-Aware Sales Forecasting
--------------------------------------------------------------
Generates:
- Temporal Lag features (lag_1, lag_2, lag_4, lag_8)
- Rolling window statistics (rolling_mean_4, rolling_std_4, rolling_mean_8, rolling_mean_12)
- Cyclical calendar features (sine/cosine week and month encodings, festival season flag)
- Multi-domain interaction features (Disaster Severity x Industry Sensitivity, Supply x Channel, Survey x Channel)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the aligned sales dataframe and creates time-series lags, rolling aggregations,
    cyclical temporal features, and disaster-business interaction features.
    """
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['region', 'category', 'channel', 'date']).reset_index(drop=True)

    # 1. Cyclical Calendar Features
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_year'] / 52.0)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_year'] / 52.0)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)
    data['is_festival_season'] = data['month'].apply(lambda m: 1 if m in [10, 11] else 0)
    data['is_harvest_season'] = data.apply(
        lambda r: 1 if (r['month'] in [3, 4, 10, 11] and 'Farming' in r['category']) else 0,
        axis=1
    )

    # 2. Channel binary indicators
    data['is_offline'] = data['channel'].apply(lambda c: 1 if 'Offline' in c else 0)
    data['is_online'] = 1 - data['is_offline']

    # 3. Time-Series Lag Features (grouped by series key)
    group_cols = ['region', 'category', 'channel']
    
    for lag in [1, 2, 4, 8]:
        data[f'lag_rev_{lag}'] = data.groupby(group_cols)['revenue_inr'].shift(lag)
        data[f'lag_units_{lag}'] = data.groupby(group_cols)['units_sold'].shift(lag)

    # 4. Rolling Window Statistics
    for window in [4, 8, 12]:
        # Using shift(1) to strictly avoid data leakage of the current week's actual sales
        data[f'rolling_mean_rev_{window}'] = (
            data.groupby(group_cols)['revenue_inr']
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        data[f'rolling_std_rev_{window}'] = (
            data.groupby(group_cols)['revenue_inr']
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )
        data[f'rolling_mean_units_{window}'] = (
            data.groupby(group_cols)['units_sold']
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    # 5. Multi-Domain Disaster & Business Interaction Features
    # Disaster Severity x Industry Sensitivity
    data['disaster_x_industry_interaction'] = data['disaster_severity'] * data['industry_sensitivity']
    
    # Disaster Severity x Duration Persistence
    data['disaster_x_duration'] = data['disaster_severity'] * (data['disaster_duration_days'] / 30.0)
    
    # Supply Disruption x Channel Vulnerability
    data['supply_x_offline_shock'] = data['supply_disruption_index'] * data['is_offline']
    
    # Mobility Drop x Offline Store Impact
    data['mobility_x_offline_shock'] = (data['mobility_reduction_pct'] / 100.0) * data['is_offline']
    
    # Survey Online Channel Shift Signal x Online Channel Boomer
    data['survey_shift_x_online'] = (data['survey_channel_shift_pct'] / 100.0) * data['is_online']
    
    # Agri Disruption x Farming Vulnerability
    data['agri_disruption_interaction'] = data.apply(
        lambda r: r['disaster_severity'] * 1.5 if ('Farming' in str(r['category']) or 'Food' in str(r['category'])) else 0.0,
        axis=1
    )

    # AI Job Recession x Tech / Discretionary Vulnerability
    data['ai_recession_tech_shock'] = data.apply(
        lambda r: (r['economic_disruption_index'] * 2.0) if (
            'AI' in str(r['disaster_type']) and ('Tech' in str(r['category']) or 'Luxury' in str(r['category']) or 'Hospitality' in str(r['category']))
        ) else 0.0,
        axis=1
    )

    # Fill initial lag NaNs with backfill/first valid value
    feature_cols_with_na = [c for c in data.columns if 'lag_' in c or 'rolling_' in c]
    data[feature_cols_with_na] = data.groupby(group_cols)[feature_cols_with_na].bfill()
    data[feature_cols_with_na] = data[feature_cols_with_na].fillna(0)

    return data
