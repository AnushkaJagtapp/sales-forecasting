"""
Disaster Impact Score Calculation Module
----------------------------------------
Computes a transparent, defensible 0-100 Disaster Impact Score using a multi-factor
normalized weighted composite formula, along with risk categorization and industry/event sensitivity.
Supports natural disasters, pandemics, traditional farming crop failures/monsoon shocks,
and modern AI job recession / economic white-collar disruption.
"""

from typing import Dict, Any, Tuple

# Defensible weights for Disaster Impact Score (Sum = 1.00)
IMPACT_WEIGHTS = {
    'severity': 0.25,           # Direct operational shock intensity
    'supply_disruption': 0.25,  # Inventory shortage, crop yield failure & logistics blockage
    'duration': 0.15,           # Temporal persistence of the disruption event
    'economic_disruption': 0.15,# Macro demand, purchasing power shock & layoff sentiment
    'geographic_spread': 0.10,  # Regional reach (localized district vs nationwide)
    'population_affected': 0.10 # Consumer / workforce base density impacted
}

# Industry / Product Category Sensitivity Multipliers
INDUSTRY_SENSITIVITY = {
    'Traditional Farming & Agri Produce': 1.30,   # Highly vulnerable to monsoon/climate/pests
    'Luxury Goods & Premium Apparel': 1.35,       # Discretionary cut during AI layoffs/recession
    'Restaurant Dine-in & Hospitality': 1.40,     # Heavily impacted during lockdowns & mobility drops
    'Tourism & Travel': 1.45,                     # First to be slashed during crises
    'Electronics & Workstation Tech': 1.10,       # Mixed: surge in WFH tech, but slump during layoffs
    'General Retail & FMCG': 1.00,                # Baseline sensitivity
    'Personal Care & Cosmetics': 0.85,            # Stable essentials, minor premium cuts
    'Packaged Food & Beverages': 0.55,            # Resilient staple demand
    'Fresh Foods & Grocery': 0.45,                # Inelastic essential demand
    'Medical Supplies & Healthcare': 0.30,        # Anti-cyclical / surge during health crises
    'Digital Services & AI Upskilling': 0.50      # High demand during job transitions/layoffs
}

# Event Types and their specific vulnerability characteristics
DISASTER_TYPES = [
    'Pandemic (COVID-19)',
    'Traditional Farming Crop Failure / Monsoon Drought',
    'Severe Flood / Konkan Inundation',
    'Cyclone (Amphan / Fani / Biparjoy)',
    'Modern AI Job Recession & Tech Layoffs',
    'Global & Domestic Supply Chain Bottleneck'
]

def calculate_disaster_impact_score(
    severity: float,              # Scale 0 - 10
    duration_days: float,         # Days (0 to 180+)
    geographic_spread: float,     # Scale 0 - 1 (or 1 to 5 converted to 0-1)
    population_affected: float,   # Scale 0 - 1 (fraction of population impacted)
    economic_disruption: float,   # Scale 0 - 1
    supply_disruption: float,     # Scale 0 - 1
    industry: str = None
) -> Tuple[float, str, Dict[str, float]]:
    """
    Computes the composite 0-100 Disaster Impact Score, its risk category,
    and the component breakdown.

    Parameters:
    -----------
    severity: float (0 to 10)
    duration_days: float (days)
    geographic_spread: float (0.0 to 1.0)
    population_affected: float (0.0 to 1.0)
    economic_disruption: float (0.0 to 1.0)
    supply_disruption: float (0.0 to 1.0)
    industry: optional string to apply category-specific sensitivity weighting

    Returns:
    --------
    (score, risk_level, components_breakdown)
    """
    # 1. Normalize severity (0 to 10 -> 0.0 to 1.0)
    s_norm = max(0.0, min(1.0, float(severity) / 10.0))
    
    # 2. Normalize duration (bounded at 120 days max for saturation)
    d_norm = max(0.0, min(1.0, float(duration_days) / 120.0))
    
    # 3. Ensure other components bounded [0, 1]
    g_norm = max(0.0, min(1.0, float(geographic_spread)))
    p_norm = max(0.0, min(1.0, float(population_affected)))
    e_norm = max(0.0, min(1.0, float(economic_disruption)))
    sc_norm = max(0.0, min(1.0, float(supply_disruption)))

    # 4. Weighted Base Score (0 to 100)
    base_score = (
        IMPACT_WEIGHTS['severity'] * s_norm +
        IMPACT_WEIGHTS['supply_disruption'] * sc_norm +
        IMPACT_WEIGHTS['duration'] * d_norm +
        IMPACT_WEIGHTS['economic_disruption'] * e_norm +
        IMPACT_WEIGHTS['geographic_spread'] * g_norm +
        IMPACT_WEIGHTS['population_affected'] * p_norm
    ) * 100.0

    # 5. Industry-adjusted score
    sensitivity = INDUSTRY_SENSITIVITY.get(industry, 1.0) if industry else 1.0
    final_score = round(max(0.0, min(100.0, base_score * sensitivity)), 2)

    # 6. Risk Level Categorization
    if final_score <= 25.0:
        risk_level = "Low"
    elif final_score <= 50.0:
        risk_level = "Moderate"
    elif final_score <= 75.0:
        risk_level = "High"
    else:
        risk_level = "Critical"

    breakdown = {
        'Severity Shock': round(s_norm * IMPACT_WEIGHTS['severity'] * 100.0, 2),
        'Supply Disruption': round(sc_norm * IMPACT_WEIGHTS['supply_disruption'] * 100.0, 2),
        'Duration Impact': round(d_norm * IMPACT_WEIGHTS['duration'] * 100.0, 2),
        'Economic Disruption': round(e_norm * IMPACT_WEIGHTS['economic_disruption'] * 100.0, 2),
        'Geographic Spread': round(g_norm * IMPACT_WEIGHTS['geographic_spread'] * 100.0, 2),
        'Population Affected': round(p_norm * IMPACT_WEIGHTS['population_affected'] * 100.0, 2),
        'Raw Base Score': round(base_score, 2),
        'Industry Sensitivity': sensitivity,
        'Final Impact Score': final_score
    }

    return final_score, risk_level, breakdown
