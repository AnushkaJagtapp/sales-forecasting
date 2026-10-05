"""
Data Preprocessing and Validation Module
----------------------------------------
Provides robust schema validation, missing value imputation, duplicate removal,
data-type conversion, date alignment, disaster log merging, and out-of-time
train/test splitting to prevent future data leakage.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder

# Standard column aliases for flexible user CSV uploads
COLUMN_ALIASES = {
    'date': ['date', 'Date', 'DATE', 'order_date', 'Order_Date', 'timestamp', 'week_date', 'sales_date'],
    'revenue_inr': ['revenue_inr', 'revenue', 'Revenue', 'sales', 'Sales', 'total_sales', 'Total_Sales', 'amount', 'Amount'],
    'units_sold': ['units_sold', 'units', 'Units', 'quantity', 'Quantity', 'qty', 'volume', 'sales_volume'],
    'region': ['region', 'Region', 'state', 'State', 'zone', 'Zone', 'market', 'location'],
    'category': ['category', 'Category', 'product_category', 'Product_Category', 'sector', 'industry', 'item_category'],
    'channel': ['channel', 'Channel', 'fulfillment_channel', 'sales_channel', 'Channel_Type', 'platform']
}

DISASTER_ALIASES = {
    'disaster_name': ['disaster_name', 'name', 'event_name', 'Disaster_Name', 'event', 'Event'],
    'disaster_type': ['disaster_type', 'type', 'Type', 'category', 'Disaster_Type', 'event_type'],
    'start_date': ['start_date', 'start', 'Start_Date', 'from_date', 'begin_date'],
    'end_date': ['end_date', 'end', 'End_Date', 'to_date', 'finish_date'],
    'severity': ['severity', 'Severity', 'impact_level', 'severity_score', 'magnitude'],
    'affected_regions': ['affected_regions', 'regions', 'affected_region', 'location', 'states']
}

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


def validate_sales_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    Validates uploaded sales CSV against required schema with alias mapping.

    Returns:
    --------
    (is_valid, messages, mapped_columns)
    """
    messages = []
    mapped_columns = {}
    
    if df is None or df.empty:
        return False, ["Uploaded sales file is empty."], {}

    df_cols = list(df.columns)
    df_cols_lower = {c.lower().strip(): c for c in df_cols}

    # Match mandatory columns
    required_keys = ['date', 'revenue_inr']
    for req in required_keys:
        matched = None
        for alias in COLUMN_ALIASES[req]:
            if alias.lower() in df_cols_lower:
                matched = df_cols_lower[alias.lower()]
                break
        if matched:
            mapped_columns[req] = matched
        else:
            messages.append(f"Missing required column for '{req}'. Accepted aliases: {', '.join(COLUMN_ALIASES[req])}")

    # Match optional columns
    optional_keys = ['units_sold', 'region', 'category', 'channel']
    for opt in optional_keys:
        matched = None
        for alias in COLUMN_ALIASES[opt]:
            if alias.lower() in df_cols_lower:
                matched = df_cols_lower[alias.lower()]
                break
        if matched:
            mapped_columns[opt] = matched
        else:
            messages.append(f"Notice: Optional column '{opt}' not found. Default values will be generated automatically.")

    is_valid = len([m for m in messages if m.startswith("Missing required")]) == 0
    return is_valid, messages, mapped_columns


def validate_disaster_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    Validates uploaded disaster events CSV against expected schema with alias mapping.
    """
    messages = []
    mapped_columns = {}

    if df is None or df.empty:
        return False, ["Disaster file is empty."], {}

    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    
    for key, aliases in DISASTER_ALIASES.items():
        matched = None
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                matched = df_cols_lower[alias.lower()]
                break
        if matched:
            mapped_columns[key] = matched
        else:
            if key in ['start_date', 'end_date', 'severity']:
                messages.append(f"Missing core disaster column '{key}'. Accepted aliases: {', '.join(aliases)}")

    is_valid = len([m for m in messages if m.startswith("Missing core")]) == 0
    return is_valid, messages, mapped_columns


def preprocess_sales_data(
    df: pd.DataFrame,
    disaster_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans, standardizes, imputes, and preprocesses raw sales and optional disaster data.

    Returns:
    --------
    (cleaned_df, report_dict)
    """
    report = {
        'initial_rows': len(df),
        'missing_values_handled': 0,
        'duplicates_removed': 0,
        'date_errors_handled': 0,
        'actions': []
    }

    cleaned = df.copy()

    # 1. Standardize column names via aliases
    _, _, mapped_cols = validate_sales_data(cleaned)
    rename_map = {v: k for k, v in mapped_cols.items()}
    cleaned = cleaned.rename(columns=rename_map)

    # 2. Date parsing & conversion
    if 'date' in cleaned.columns:
        initial_dates = len(cleaned)
        cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')
        invalid_dates = cleaned['date'].isna().sum()
        if invalid_dates > 0:
            cleaned = cleaned.dropna(subset=['date'])
            report['date_errors_handled'] = int(invalid_dates)
            report['actions'].append(f"Dropped {invalid_dates} rows with unparseable dates.")

    # 3. Numeric Conversions & Missing Values
    if 'revenue_inr' in cleaned.columns:
        cleaned['revenue_inr'] = pd.to_numeric(cleaned['revenue_inr'], errors='coerce')
        na_rev = cleaned['revenue_inr'].isna().sum()
        if na_rev > 0:
            median_rev = cleaned['revenue_inr'].median()
            cleaned['revenue_inr'] = cleaned['revenue_inr'].fillna(median_rev if not np.isnan(median_rev) else 1000.0)
            report['missing_values_handled'] += int(na_rev)
            report['actions'].append(f"Imputed {na_rev} missing revenue records with median value.")

    if 'units_sold' not in cleaned.columns:
        cleaned['units_sold'] = np.maximum(1, (cleaned['revenue_inr'] / 150.0).round().astype(int))
        report['actions'].append("Generated default 'units_sold' from revenue values.")
    else:
        cleaned['units_sold'] = pd.to_numeric(cleaned['units_sold'], errors='coerce').fillna(1).astype(int)

    # 4. Fill optional categoricals
    if 'region' not in cleaned.columns:
        cleaned['region'] = 'National Market'
        report['actions'].append("Assigned default 'National Market' to missing region.")
    else:
        cleaned['region'] = cleaned['region'].fillna('Unknown Region').astype(str)

    if 'category' not in cleaned.columns:
        cleaned['category'] = 'General Retail & FMCG'
        report['actions'].append("Assigned default 'General Retail & FMCG' to missing category.")
    else:
        cleaned['category'] = cleaned['category'].fillna('General Retail & FMCG').astype(str)

    if 'channel' not in cleaned.columns:
        cleaned['channel'] = 'Offline Store / Mandi'
        report['actions'].append("Assigned default 'Offline Store / Mandi' to missing channel.")
    else:
        cleaned['channel'] = cleaned['channel'].fillna('Offline Store / Mandi').astype(str)

    # 5. Deduplication
    initial_len = len(cleaned)
    dedup_subset = ['date', 'region', 'category', 'channel']
    existing_subset = [c for c in dedup_subset if c in cleaned.columns]
    if existing_subset:
        cleaned = cleaned.drop_duplicates(subset=existing_subset)
        dups_dropped = initial_len - len(cleaned)
        report['duplicates_removed'] = int(dups_dropped)
        if dups_dropped > 0:
            report['actions'].append(f"Removed {dups_dropped} duplicate time-series records.")

    # 6. Temporal Features
    cleaned['date'] = pd.to_datetime(cleaned['date'])
    cleaned['year'] = cleaned['date'].dt.year
    cleaned['month'] = cleaned['date'].dt.month
    cleaned['quarter'] = cleaned['date'].dt.quarter
    cleaned['week_of_year'] = cleaned['date'].dt.isocalendar().week.astype(int)

    # 7. Merge Disaster Events if supplied
    if disaster_df is not None and not disaster_df.empty:
        cleaned = _merge_disaster_logs(cleaned, disaster_df, report)
    else:
        # Guarantee disaster columns exist for downstream ML
        for col, default_val in [
            ('is_disaster_active', 0), ('disaster_name', 'Normal Conditions'),
            ('disaster_type', 'None'), ('disaster_severity', 0.0),
            ('disaster_duration_days', 0.0), ('supply_disruption_index', 0.0),
            ('economic_disruption_index', 0.0), ('geographic_spread_index', 0.0),
            ('population_affected_index', 0.0), ('mobility_reduction_pct', 0.0),
            ('survey_channel_shift_pct', 0.0), ('industry_sensitivity', 1.0),
            ('disaster_impact_score', 0.0), ('risk_level', 'Low')
        ]:
            if col not in cleaned.columns:
                cleaned[col] = default_val

    # Ensure price & discount
    if 'price' not in cleaned.columns:
        cleaned['price'] = (cleaned['revenue_inr'] / (cleaned['units_sold'] + 1e-6)).round(2)
    if 'discount_pct' not in cleaned.columns:
        cleaned['discount_pct'] = 0.05

    report['final_rows'] = len(cleaned)
    report['min_date'] = cleaned['date'].min().strftime('%Y-%m-%d') if len(cleaned) > 0 else 'N/A'
    report['max_date'] = cleaned['date'].max().strftime('%Y-%m-%d') if len(cleaned) > 0 else 'N/A'
    report['unique_regions'] = cleaned['region'].nunique()
    report['unique_categories'] = cleaned['category'].nunique()

    return cleaned, report


def _merge_disaster_logs(sales_df: pd.DataFrame, disaster_df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    """Helper to merge external disaster event records into sales timeseries."""
    df = sales_df.copy()
    d_df = disaster_df.copy()

    _, _, mapped_d = validate_disaster_data(d_df)
    d_df = d_df.rename(columns={v: k for k, v in mapped_d.items()})

    if 'start_date' in d_df.columns:
        d_df['start_date'] = pd.to_datetime(d_df['start_date'], errors='coerce')
    if 'end_date' in d_df.columns:
        d_df['end_date'] = pd.to_datetime(d_df['end_date'], errors='coerce')

    # Initialize default shock columns
    if 'is_disaster_active' not in df.columns:
        df['is_disaster_active'] = 0
        df['disaster_name'] = 'Normal Conditions'
        df['disaster_type'] = 'None'
        df['disaster_severity'] = 0.0
        df['disaster_duration_days'] = 0.0
        df['supply_disruption_index'] = 0.0
        df['economic_disruption_index'] = 0.0
        df['geographic_spread_index'] = 0.0
        df['population_affected_index'] = 0.0
        df['mobility_reduction_pct'] = 0.0
        df['survey_channel_shift_pct'] = 0.0
        df['disaster_impact_score'] = 0.0
        df['risk_level'] = 'Low'

    merged_events_count = 0
    for _, row in d_df.iterrows():
        s_date = row.get('start_date')
        e_date = row.get('end_date')
        if pd.isna(s_date) or pd.isna(e_date):
            continue

        mask = (df['date'] >= s_date) & (df['date'] <= e_date)
        aff_reg = str(row.get('affected_regions', 'All India'))
        if aff_reg != 'All India' and 'all' not in aff_reg.lower():
            regs = [r.strip() for r in aff_reg.split(',')]
            mask = mask & (df['region'].isin(regs))

        if mask.sum() > 0:
            merged_events_count += 1
            df.loc[mask, 'is_disaster_active'] = 1
            df.loc[mask, 'disaster_name'] = str(row.get('disaster_name', 'Disaster Event'))
            df.loc[mask, 'disaster_type'] = str(row.get('disaster_type', 'General Shock'))
            df.loc[mask, 'disaster_severity'] = float(row.get('severity', 5.0))
            df.loc[mask, 'disaster_duration_days'] = float(row.get('duration_days', (e_date - s_date).days))
            df.loc[mask, 'supply_disruption_index'] = float(row.get('supply_disruption', 0.5))
            df.loc[mask, 'economic_disruption_index'] = float(row.get('economic_disruption', 0.5))
            df.loc[mask, 'geographic_spread_index'] = float(row.get('geographic_spread', 0.5))
            df.loc[mask, 'population_affected_index'] = float(row.get('population_affected', 0.5))
            df.loc[mask, 'mobility_reduction_pct'] = float(row.get('mobility_reduction_pct', 20.0))

    report['actions'].append(f"Successfully integrated {merged_events_count} disaster events into sales timeline.")
    return df


def prepare_and_preprocess_data(
    data_path: str = "data/sample_sales.csv",
    split_year: int = 2023,
    models_dir: str = "models"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    """
    Loads dataset, applies feature engineering, encodes categoricals, and splits data by time.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Check fallback paths
    if not os.path.exists(data_path):
        alt_paths = ["data/processed/final_dataset.csv", "data/raw/historical_sales.csv"]
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                break

    df_raw = pd.read_csv(data_path)
    
    # Import create_features safely
    from src.features import create_features
    df_featured = create_features(df_raw)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        if col in df_featured.columns:
            df_featured[f"{col}_encoded"] = le.fit_transform(df_featured[col].astype(str))
        else:
            df_featured[f"{col}_encoded"] = 0
            le.fit(['Default'])
        encoders[col] = le
        joblib.dump(le, os.path.join(models_dir, f"{col}_encoder.pkl"))

    # Time-based Train / Test Split (Strict zero-leakage)
    if 'year' not in df_featured.columns:
        df_featured['year'] = pd.to_datetime(df_featured['date']).dt.year

    train_df = df_featured[df_featured['year'] < split_year].copy()
    test_df = df_featured[df_featured['year'] >= split_year].copy()

    # Fallback split if years are missing or single year
    if len(train_df) == 0 or len(test_df) == 0:
        split_idx = int(len(df_featured) * 0.8)
        train_df = df_featured.iloc[:split_idx].copy()
        test_df = df_featured.iloc[split_idx:].copy()

    # Ensure all feature columns exist
    for f in FEATURE_COLUMNS:
        if f not in train_df.columns:
            train_df[f] = 0.0
            test_df[f] = 0.0

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    joblib.dump(FEATURE_COLUMNS, os.path.join(models_dir, "feature_columns.pkl"))
    return X_train, y_train, X_test, y_test, encoders, FEATURE_COLUMNS
