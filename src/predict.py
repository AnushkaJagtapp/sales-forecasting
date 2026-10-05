"""
Inference & Prediction Pipeline for Disaster-Aware Sales Forecasting
---------------------------------------------------------------------
Handles single-scenario simulation, batch prediction, impact quantification,
risk categorization, recovery estimation, and SHAP explainability.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.impact_score import calculate_disaster_impact_score, INDUSTRY_SENSITIVITY
from src.recovery import estimate_scenario_recovery_weeks
from src.explain import explain_prediction

MODELS_DIR = "models"

def load_prediction_artifacts(models_dir: str = MODELS_DIR):
    """Loads trained model, encoders, and feature column list."""
    best_model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    
    encoders = {}
    for col in ['region', 'category', 'channel', 'disaster_type']:
        enc_path = os.path.join(models_dir, f"{col}_encoder.pkl")
        if os.path.exists(enc_path):
            encoders[col] = joblib.load(enc_path)
            
    return best_model, feature_cols, encoders

def predict_scenario(
    region: str,
    category: str,
    channel: str,
    disaster_type: str,
    disaster_severity: float,
    disaster_duration_days: float,
    supply_disruption_index: float,
    economic_disruption_index: float,
    geographic_spread_index: float,
    population_affected_index: float,
    mobility_reduction_pct: float,
    baseline_revenue: float,
    month: int = 6,
    week_of_year: int = 24,
    models_dir: str = MODELS_DIR,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end inference on a user-defined disaster & business scenario.
    """
    best_model, feature_cols, encoders = load_prediction_artifacts(models_dir)
    
    # If specific model selected
    if model_name:
        slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        custom_model_path = os.path.join(models_dir, f"{slug}.pkl")
        if os.path.exists(custom_model_path):
            best_model = joblib.load(custom_model_path)

    # 1. Compute Disaster Impact Score & Risk
    impact_score, risk_level, score_breakdown = calculate_disaster_impact_score(
        severity=disaster_severity,
        duration_days=disaster_duration_days,
        geographic_spread=geographic_spread_index,
        population_affected=population_affected_index,
        economic_disruption=economic_disruption_index,
        supply_disruption=supply_disruption_index,
        industry=category
    )

    # 2. Encode Categoricals
    def safe_encode(enc, val):
        if val in enc.classes_:
            return int(enc.transform([val])[0])
        return 0

    region_enc = safe_encode(encoders['region'], region)
    category_enc = safe_encode(encoders['category'], category)
    channel_enc = safe_encode(encoders['channel'], channel)
    disaster_type_enc = safe_encode(encoders['disaster_type'], disaster_type)

    is_offline = 1 if 'Offline' in channel else 0
    is_online = 1 - is_offline
    industry_sens = INDUSTRY_SENSITIVITY.get(category, 1.0)

    # Default category pricing
    cat_price_map = {
        'Traditional Farming & Agri Produce': 45.0,
        'Electronics & Workstation Tech': 16500.0,
        'Packaged Food & Beverages': 190.0,
        'Personal Care & Cosmetics': 420.0,
        'Fresh Foods & Grocery': 85.0,
        'Medical Supplies & Healthcare': 650.0,
        'Restaurant Dine-in & Hospitality': 850.0,
        'Luxury Goods & Premium Apparel': 4500.0,
        'Digital Services & AI Upskilling': 2500.0
    }
    price = cat_price_map.get(category, 500.0)

    # Survey channel shift default signal
    survey_shift_map = {
        'Traditional Farming & Agri Produce': 42.0,
        'Electronics & Workstation Tech': 55.0,
        'Packaged Food & Beverages': 48.0,
        'Personal Care & Cosmetics': 55.0,
        'Fresh Foods & Grocery': 46.0,
        'Medical Supplies & Healthcare': 68.0,
        'Restaurant Dine-in & Hospitality': -45.0,
        'Luxury Goods & Premium Apparel': -30.0,
        'Digital Services & AI Upskilling': 72.0
    }
    survey_shift = survey_shift_map.get(category, 10.0)

    # 3. Assemble Feature Vector
    is_disaster = 1 if disaster_severity > 0.5 and disaster_type != 'None' else 0
    
    feature_dict = {
        'region_encoded': region_enc,
        'category_encoded': category_enc,
        'channel_encoded': channel_enc,
        'disaster_type_encoded': disaster_type_enc,
        'month': month,
        'quarter': (month - 1) // 3 + 1,
        'week_of_year': week_of_year,
        'week_sin': np.sin(2 * np.pi * week_of_year / 52.0),
        'week_cos': np.cos(2 * np.pi * week_of_year / 52.0),
        'month_sin': np.sin(2 * np.pi * month / 12.0),
        'month_cos': np.cos(2 * np.pi * month / 12.0),
        'is_festival_season': 1 if month in [10, 11] else 0,
        'is_harvest_season': 1 if (month in [3, 4, 10, 11] and 'Farming' in category) else 0,
        'is_offline': is_offline,
        'is_online': is_online,
        'price': price,
        'discount_pct': 0.08 if is_online else 0.05,
        'industry_sensitivity': industry_sens,
        'is_disaster_active': is_disaster,
        'disaster_severity': disaster_severity,
        'disaster_duration_days': disaster_duration_days,
        'supply_disruption_index': supply_disruption_index,
        'economic_disruption_index': economic_disruption_index,
        'geographic_spread_index': geographic_spread_index,
        'population_affected_index': population_affected_index,
        'mobility_reduction_pct': mobility_reduction_pct,
        'survey_channel_shift_pct': survey_shift,
        'disaster_impact_score': impact_score,
        'disaster_x_industry_interaction': disaster_severity * industry_sens,
        'disaster_x_duration': disaster_severity * (disaster_duration_days / 30.0),
        'supply_x_offline_shock': supply_disruption_index * is_offline,
        'mobility_x_offline_shock': (mobility_reduction_pct / 100.0) * is_offline,
        'survey_shift_x_online': (survey_shift / 100.0) * is_online,
        'agri_disruption_interaction': (disaster_severity * 1.5) if ('Farming' in category or 'Food' in category) else 0.0,
        'ai_recession_tech_shock': (economic_disruption_index * 2.0) if ('AI' in disaster_type and ('Tech' in category or 'Luxury' in category or 'Hospitality' in category)) else 0.0,
        
        # Approximate baseline lag signals
        'lag_rev_1': baseline_revenue * 0.98,
        'lag_rev_2': baseline_revenue * 0.99,
        'lag_rev_4': baseline_revenue * 1.01,
        'lag_rev_8': baseline_revenue * 1.00,
        'lag_units_1': max(1, int(baseline_revenue / price)),
        'lag_units_2': max(1, int(baseline_revenue / price)),
        'lag_units_4': max(1, int(baseline_revenue / price)),
        'lag_units_8': max(1, int(baseline_revenue / price)),
        'rolling_mean_rev_4': baseline_revenue,
        'rolling_std_rev_4': baseline_revenue * 0.05,
        'rolling_mean_rev_8': baseline_revenue,
        'rolling_mean_rev_12': baseline_revenue,
        'rolling_mean_units_4': max(1, int(baseline_revenue / price))
    }

    df_row = pd.DataFrame([feature_dict])[feature_cols]

    # 4. Predict
    pred_sales = float(best_model.predict(df_row)[0])
    pred_sales = max(0.0, pred_sales)

    # 5. Impact %
    impact_pct = round(((pred_sales - baseline_revenue) / (baseline_revenue + 1e-9)) * 100.0, 2)

    # 6. Recovery Estimate
    recovery_weeks = estimate_scenario_recovery_weeks(
        disaster_severity=disaster_severity,
        duration_days=disaster_duration_days,
        industry_sensitivity=industry_sens,
        is_online=(is_online == 1)
    )

    # 7. Explainability
    explanation = explain_prediction(best_model, df_row, feature_cols)

    return {
        'baseline_revenue': baseline_revenue,
        'predicted_revenue': round(pred_sales, 2),
        'impact_percentage': impact_pct,
        'revenue_change_inr': round(pred_sales - baseline_revenue, 2),
        'impact_score': impact_score,
        'risk_level': risk_level,
        'estimated_recovery_weeks': recovery_weeks,
        'score_breakdown': score_breakdown,
        'explanation': explanation,
        'feature_vector': feature_dict
    }
