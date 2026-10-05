"""
Disaster Recovery & Before-During-After Analysis Module
------------------------------------------------------
Performs:
1. Phase segmentation (Before, During, After disruption event)
2. Baseline sales computation
3. Actual recovery crossing threshold analysis (95% recovery threshold)
4. Empirical scenario recovery timeline estimation for What-If simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def analyze_disaster_phases(
    df: pd.DataFrame,
    disaster_start: str,
    disaster_end: str,
    region: Optional[str] = None,
    category: Optional[str] = None,
    channel: Optional[str] = None,
    recovery_threshold_pct: float = 0.95,
    baseline_window_weeks: int = 6
) -> Dict[str, Any]:
    """
    Performs Before-During-After phase analysis on historical sales data.

    Parameters:
    -----------
    df: historical sales dataframe with 'date' and 'revenue_inr'
    disaster_start: start date string 'YYYY-MM-DD'
    disaster_end: end date string 'YYYY-MM-DD'
    region, category, channel: optional filters
    recovery_threshold_pct: threshold to define recovery (e.g. 0.95 = 95% of baseline)
    baseline_window_weeks: number of weeks prior to start date used to establish normal baseline
    """
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    
    if region and region != 'All Regions':
        data = data[data['region'] == region]
    if category and category != 'All Categories':
        data = data[data['category'] == category]
    if channel and channel != 'All Channels':
        data = data[data['channel'] == channel]

    # Aggregate weekly revenue across filtered segments
    weekly = data.groupby('date')['revenue_inr'].sum().reset_index().sort_values('date')
    
    start_dt = pd.to_datetime(disaster_start)
    end_dt = pd.to_datetime(disaster_end)
    baseline_start_dt = start_dt - pd.Timedelta(weeks=baseline_window_weeks)

    # 1. Phase 1: Before Disaster
    before_df = weekly[(weekly['date'] >= baseline_start_dt) & (weekly['date'] < start_dt)]
    baseline_revenue = before_df['revenue_inr'].mean() if len(before_df) > 0 else weekly['revenue_inr'].mean()

    # 2. Phase 2: During Disaster
    during_df = weekly[(weekly['date'] >= start_dt) & (weekly['date'] <= end_dt)]
    during_revenue = during_df['revenue_inr'].mean() if len(during_df) > 0 else baseline_revenue

    # 3. Phase 3: After Disaster
    after_df = weekly[weekly['date'] > end_dt].copy().reset_index(drop=True)

    # Calculate Impact %
    impact_pct = round(((during_revenue - baseline_revenue) / baseline_revenue) * 100.0, 2) if baseline_revenue > 0 else 0.0

    # 4. Recovery Timeline Calculation
    target_recovery_rev = baseline_revenue * recovery_threshold_pct
    recovery_weeks = None
    recovery_date = None

    for i, row in after_df.iterrows():
        if row['revenue_inr'] >= target_recovery_rev:
            recovery_weeks = i + 1
            recovery_date = row['date'].strftime('%Y-%m-%d')
            break

    if recovery_weeks is None:
        recovery_status = f"> {len(after_df)} weeks (Ongoing / Extended)"
        est_recovery_weeks = len(after_df) + 2
    else:
        recovery_status = f"{recovery_weeks} weeks ({recovery_date})"
        est_recovery_weeks = recovery_weeks

    # Total estimated revenue loss during disruption window
    expected_during_rev = baseline_revenue * len(during_df)
    actual_during_rev = during_df['revenue_inr'].sum()
    revenue_delta = round(actual_during_rev - expected_during_rev, 2)

    return {
        'baseline_weekly_revenue': round(baseline_revenue, 2),
        'during_weekly_revenue': round(during_revenue, 2),
        'impact_percentage': impact_pct,
        'revenue_deviation_inr': revenue_delta,
        'recovery_threshold_inr': round(target_recovery_rev, 2),
        'recovery_status': recovery_status,
        'recovery_weeks': est_recovery_weeks,
        'before_records_count': len(before_df),
        'during_records_count': len(during_df),
        'after_records_count': len(after_df)
    }


def estimate_scenario_recovery_weeks(
    disaster_severity: float,
    duration_days: float,
    industry_sensitivity: float,
    is_online: bool = False
) -> int:
    """
    Empirical decision-support function to estimate recovery timeline (in weeks)
    under custom simulated disaster scenarios.
    """
    if disaster_severity <= 1.0:
        return 0
    
    # Base weeks calculation
    base_weeks = (disaster_severity * 0.8) + (duration_days / 20.0)
    adjusted = base_weeks * industry_sensitivity

    # Online / D2C channels recover faster
    if is_online:
        adjusted *= 0.65

    return max(1, int(round(adjusted)))
