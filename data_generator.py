"""
Comprehensive Data Generator for Disaster-Aware Sales Forecasting System
-----------------------------------------------------------------------
Generates:
1. Real-calibrated BCG India COVID-19 Consumer Sentiment Survey Wave 1 (N = 2,106)
2. Comprehensive Disruption & Disaster Event Registry (Pandemic, Floods, Cyclones,
   Traditional Farming Crop Failure / Monsoon Drought, Modern AI Job Recession & Layoffs, Supply Bottlenecks)
3. Multi-year weekly retail and agricultural sales dataset (2019 - 2023) across 6 regions, 8 product sectors,
   and Offline vs Online/D2C channels.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.impact_score import calculate_disaster_impact_score, INDUSTRY_SENSITIVITY

np.random.seed(42)

def generate_bcg_survey_data(output_path: str = "data/raw/survey_bcg_covid19.csv"):
    """
    Generates BCG India COVID-19 Consumer Sentiment Survey Wave 1 (March 23-26, 2020).
    Total respondents: N = 2,106.
    Reflects reported category online spending expectations & sentiment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = 2106

    respondent_ids = [f"RESP_{i:04d}" for i in range(1, n + 1)]
    age_groups = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=n, p=[0.20, 0.35, 0.25, 0.12, 0.08])
    cities = np.random.choice(['Mumbai', 'Delhi NCR', 'Bengaluru', 'Pune', 'Kolkata', 'Chennai', 'Bhubaneswar', 'Ahmedabad', 'Tier-2 Agri Hubs'], size=n, p=[0.16, 0.16, 0.14, 0.10, 0.10, 0.09, 0.05, 0.08, 0.12])
    income_brackets = np.random.choice(['< 5L INR', '5L - 10L INR', '10L - 20L INR', '> 20L INR'], size=n, p=[0.28, 0.40, 0.22, 0.10])
    
    categories = [
        'Electronics & Workstation Tech', 
        'Personal Care & Cosmetics', 
        'Packaged Food & Beverages', 
        'Fresh Foods & Grocery', 
        'Medical Supplies & Healthcare',
        'Restaurant Dine-in & Hospitality',
        'Luxury Goods & Premium Apparel',
        'Traditional Farming & Agri Produce',
        'Digital Services & AI Upskilling'
    ]
    
    shift_probs = {
        'Electronics & Workstation Tech': [0.55, 0.30, 0.15],
        'Personal Care & Cosmetics': [0.55, 0.28, 0.17],
        'Packaged Food & Beverages': [0.48, 0.32, 0.20],
        'Fresh Foods & Grocery': [0.46, 0.30, 0.24],
        'Medical Supplies & Healthcare': [0.68, 0.22, 0.10],
        'Restaurant Dine-in & Hospitality': [0.08, 0.15, 0.77],
        'Luxury Goods & Premium Apparel': [0.18, 0.22, 0.60],
        'Traditional Farming & Agri Produce': [0.42, 0.35, 0.23],
        'Digital Services & AI Upskilling': [0.72, 0.20, 0.08]
    }

    records = []
    for i in range(n):
        resp_id = respondent_ids[i]
        age = age_groups[i]
        city = cities[i]
        income = income_brackets[i]
        
        chosen_cats = np.random.choice(categories, size=np.random.randint(3, 7), replace=False)
        for cat in chosen_cats:
            prob = shift_probs[cat]
            sentiment_choice = np.random.choice(['Increase Online / D2C Spending', 'No Change', 'Decrease / Offline Traditional'], p=prob)
            expected_change_pct = (
                np.random.uniform(25, 75) if sentiment_choice == 'Increase Online / D2C Spending'
                else (np.random.uniform(-55, -15) if sentiment_choice == 'Decrease / Offline Traditional' else np.random.uniform(-5, 5))
            )
            records.append({
                'respondent_id': resp_id,
                'survey_wave': 'Wave 1 (March 23-26, 2020)',
                'age_group': age,
                'city': city,
                'income_bracket': income,
                'category': cat,
                'channel_preference': sentiment_choice,
                'expected_spending_shift_pct': round(expected_change_pct, 1),
                'concern_index_1_to_10': np.random.randint(6, 11)
            })

    df_survey = pd.DataFrame(records)
    df_survey.to_csv(output_path, index=False)
    print(f"Generated BCG Survey Data: {len(df_survey)} responses from {n} participants -> {output_path}")
    return df_survey


def generate_disaster_logs(output_path: str = "data/raw/disaster_events.csv"):
    """
    Generates historical disaster events in India (2019 - 2023), including:
    - Natural Disasters: Cyclones, Floods
    - Pandemics: COVID-19 Wave 1 & 2
    - Traditional Farming Disruption: Monsoon Deficit / Kharif Crop Failure
    - Modern Tech Shock: Modern AI Job Recession & Tech Layoffs Wave
    - Supply Chain Shock: Global & Domestic Logistics Bottleneck
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events = [
        {
            'event_id': 'DIS_2019_01',
            'disaster_name': 'Cyclone Fani',
            'disaster_type': 'Cyclone',
            'start_date': '2019-04-26',
            'end_date': '2019-05-18',
            'affected_regions': 'Odisha, West Bengal',
            'severity': 8.5,
            'duration_days': 22,
            'geographic_spread': 0.45,
            'population_affected': 0.35,
            'economic_disruption': 0.60,
            'supply_disruption': 0.75,
            'mobility_reduction_pct': 65.0,
            'description': 'Category 5 extremely severe cyclonic storm devastating coastal Odisha and parts of Bengal.'
        },
        {
            'event_id': 'DIS_2019_02',
            'disaster_name': 'Traditional Farming Monsoon Deficit & Crop Failure',
            'disaster_type': 'Traditional Farming Drought & Crop Shock',
            'start_date': '2019-07-01',
            'end_date': '2019-09-15',
            'affected_regions': 'Maharashtra, Gujarat, Karnataka',
            'severity': 7.2,
            'duration_days': 76,
            'geographic_spread': 0.60,
            'population_affected': 0.55,
            'economic_disruption': 0.65,
            'supply_disruption': 0.80,
            'mobility_reduction_pct': 15.0,
            'description': 'Severe rainfall deficit delaying Kharif sowing, damaging pulses and cotton yields, causing mandi price spikes.'
        },
        {
            'event_id': 'DIS_2020_01',
            'disaster_name': 'COVID-19 Wave 1 & National Lockdown',
            'disaster_type': 'Pandemic',
            'start_date': '2020-03-22',
            'end_date': '2020-07-31',
            'affected_regions': 'All India',
            'severity': 9.8,
            'duration_days': 131,
            'geographic_spread': 1.00,
            'population_affected': 0.95,
            'economic_disruption': 0.88,
            'supply_disruption': 0.92,
            'mobility_reduction_pct': 85.0,
            'description': 'Nationwide lockdown halting physical retail, closing schools and offices, surging e-commerce essentials.'
        },
        {
            'event_id': 'DIS_2020_02',
            'disaster_name': 'Super Cyclone Amphan',
            'disaster_type': 'Cyclone',
            'start_date': '2020-05-16',
            'end_date': '2020-06-05',
            'affected_regions': 'West Bengal, Odisha',
            'severity': 8.9,
            'duration_days': 21,
            'geographic_spread': 0.50,
            'population_affected': 0.40,
            'economic_disruption': 0.70,
            'supply_disruption': 0.82,
            'mobility_reduction_pct': 70.0,
            'description': 'Destructive cyclone hitting Kolkata and Sundarbans during COVID lockdown, wrecking supply corridors.'
        },
        {
            'event_id': 'DIS_2021_01',
            'disaster_name': 'COVID-19 Wave 2 (Delta Surge)',
            'disaster_type': 'Pandemic',
            'start_date': '2021-04-05',
            'end_date': '2021-06-30',
            'affected_regions': 'All India',
            'severity': 9.5,
            'duration_days': 86,
            'geographic_spread': 0.95,
            'population_affected': 0.90,
            'economic_disruption': 0.78,
            'supply_disruption': 0.82,
            'mobility_reduction_pct': 72.0,
            'description': 'Severe medical crisis leading to state-level curfews and massive surge in medical supplies and health products.'
        },
        {
            'event_id': 'DIS_2021_02',
            'disaster_name': 'Maharashtra Severe Floods & Landslides',
            'disaster_type': 'Flood',
            'start_date': '2021-07-20',
            'end_date': '2021-08-16',
            'affected_regions': 'Maharashtra',
            'severity': 8.0,
            'duration_days': 27,
            'geographic_spread': 0.35,
            'population_affected': 0.30,
            'economic_disruption': 0.55,
            'supply_disruption': 0.85,
            'mobility_reduction_pct': 60.0,
            'description': 'Record torrential monsoon rainfall flooding Chiplun, Mahad, Kolhapur and impacting Mumbai-Pune transport.'
        },
        {
            'event_id': 'DIS_2022_01',
            'disaster_name': 'Global & Domestic Supply Chain Bottleneck',
            'disaster_type': 'Supply Chain Disruption',
            'start_date': '2022-02-10',
            'end_date': '2022-04-30',
            'affected_regions': 'All India',
            'severity': 6.8,
            'duration_days': 79,
            'geographic_spread': 0.80,
            'population_affected': 0.50,
            'economic_disruption': 0.48,
            'supply_disruption': 0.88,
            'mobility_reduction_pct': 10.0,
            'description': 'Semiconductor shortages and freight cost surges delaying electronics and hardware fulfillment.'
        },
        {
            'event_id': 'DIS_2022_02',
            'disaster_name': 'Unseasonal Hailstorms & Rabi Crop Damage',
            'disaster_type': 'Traditional Farming Hail & Crop Shock',
            'start_date': '2022-03-01',
            'end_date': '2022-04-15',
            'affected_regions': 'Maharashtra, Gujarat, Delhi NCR',
            'severity': 7.0,
            'duration_days': 45,
            'geographic_spread': 0.45,
            'population_affected': 0.40,
            'economic_disruption': 0.58,
            'supply_disruption': 0.78,
            'mobility_reduction_pct': 12.0,
            'description': 'Severe hailstorms damaging standing wheat, onion, and mango crops across Western and Northern farming belts.'
        },
        {
            'event_id': 'DIS_2023_01',
            'disaster_name': 'Modern AI Job Recession & White-Collar Layoff Shock',
            'disaster_type': 'Modern AI Job Recession',
            'start_date': '2023-01-15',
            'end_date': '2023-06-30',
            'affected_regions': 'Karnataka, Maharashtra, Delhi NCR',
            'severity': 7.6,
            'duration_days': 166,
            'geographic_spread': 0.65,
            'population_affected': 0.60,
            'economic_disruption': 0.75,
            'supply_disruption': 0.20,
            'mobility_reduction_pct': 20.0,
            'description': 'GenAI automation restructuring, global tech freeze, and startup layoffs curbing discretionary tech, luxury apparel, and dining spend while boosting upskilling courses.'
        },
        {
            'event_id': 'DIS_2023_02',
            'disaster_name': 'Cyclone Biparjoy',
            'disaster_type': 'Cyclone',
            'start_date': '2023-06-06',
            'end_date': '2023-06-25',
            'affected_regions': 'Gujarat, Maharashtra',
            'severity': 7.5,
            'duration_days': 20,
            'geographic_spread': 0.40,
            'population_affected': 0.25,
            'economic_disruption': 0.50,
            'supply_disruption': 0.65,
            'mobility_reduction_pct': 45.0,
            'description': 'Severe cyclonic storm making landfall in Kutch, disrupting ports, rail corridors and local agriculture.'
        }
    ]

    df_disasters = pd.DataFrame(events)
    df_disasters.to_csv(output_path, index=False)
    print(f"Generated Disaster Logs: {len(df_disasters)} disruption events -> {output_path}")
    return df_disasters


def generate_aligned_sales_dataset(
    output_sales_path: str = "data/raw/historical_sales.csv",
    output_processed_path: str = "data/processed/final_dataset.csv"
):
    """
    Generates aligned weekly time-series sales dataset spanning 2019 to 2023 (260 weeks).
    """
    os.makedirs(os.path.dirname(output_sales_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)

    disasters_df = generate_disaster_logs()
    survey_df = generate_bcg_survey_data()

    survey_shifts = survey_df.groupby('category')['expected_spending_shift_pct'].mean().to_dict()

    start_date = datetime(2019, 1, 7) # First Monday of 2019
    end_date = datetime(2023, 12, 25) # End of 2023
    
    weeks = []
    curr = start_date
    while curr <= end_date:
        weeks.append(curr)
        curr += timedelta(days=7)

    regions = ['Maharashtra', 'Delhi NCR', 'Karnataka', 'West Bengal', 'Gujarat', 'Odisha']
    categories = [
        'Traditional Farming & Agri Produce',
        'Electronics & Workstation Tech', 
        'Packaged Food & Beverages', 
        'Personal Care & Cosmetics', 
        'Fresh Foods & Grocery', 
        'Medical Supplies & Healthcare',
        'Restaurant Dine-in & Hospitality',
        'Luxury Goods & Premium Apparel',
        'Digital Services & AI Upskilling'
    ]
    channels = ['Offline Store / Mandi', 'Online Platform / AgriTech / D2C']

    cat_base_demand = {
        'Traditional Farming & Agri Produce': {'units': 8500, 'price': 45, 'trend': 1.03},
        'Electronics & Workstation Tech': {'units': 420, 'price': 16500, 'trend': 1.05},
        'Packaged Food & Beverages': {'units': 3500, 'price': 190, 'trend': 1.03},
        'Personal Care & Cosmetics': {'units': 1900, 'price': 420, 'trend': 1.04},
        'Fresh Foods & Grocery': {'units': 5200, 'price': 85, 'trend': 1.02},
        'Medical Supplies & Healthcare': {'units': 1300, 'price': 650, 'trend': 1.06},
        'Restaurant Dine-in & Hospitality': {'units': 1500, 'price': 850, 'trend': 1.02},
        'Luxury Goods & Premium Apparel': {'units': 380, 'price': 4500, 'trend': 1.04},
        'Digital Services & AI Upskilling': {'units': 620, 'price': 2500, 'trend': 1.08}
    }

    region_multiplier = {
        'Maharashtra': 1.30,
        'Delhi NCR': 1.25,
        'Karnataka': 1.20,
        'West Bengal': 0.95,
        'Gujarat': 1.05,
        'Odisha': 0.85
    }

    rows = []

    for week in weeks:
        week_str = week.strftime('%Y-%m-%d')
        week_num = week.isocalendar()[1]
        month = week.month
        year = week.year
        quarter = (month - 1) // 3 + 1
        
        # Check active disaster for this week
        active_disaster = None
        for _, dis in disasters_df.iterrows():
            d_start = datetime.strptime(dis['start_date'], '%Y-%m-%d')
            d_end = datetime.strptime(dis['end_date'], '%Y-%m-%d')
            if d_start <= week <= d_end:
                active_disaster = dis
                break

        for region in regions:
            region_hit = False
            dis_sev = 0.0
            dis_type = 'None'
            dis_name = 'Normal Conditions'
            dis_dur = 0
            dis_supply = 0.0
            dis_econ = 0.0
            dis_geo = 0.0
            dis_pop = 0.0
            mobility_red = 0.0

            if active_disaster is not None:
                aff_regs = active_disaster['affected_regions']
                if 'All India' in aff_regs or region in aff_regs:
                    region_hit = True
                    dis_sev = float(active_disaster['severity'])
                    dis_type = active_disaster['disaster_type']
                    dis_name = active_disaster['disaster_name']
                    dis_dur = float(active_disaster['duration_days'])
                    dis_supply = float(active_disaster['supply_disruption'])
                    dis_econ = float(active_disaster['economic_disruption'])
                    dis_geo = float(active_disaster['geographic_spread'])
                    dis_pop = float(active_disaster['population_affected'])
                    mobility_red = float(active_disaster['mobility_reduction_pct'])

            for cat in categories:
                base_cfg = cat_base_demand[cat]
                reg_factor = region_multiplier[region]
                
                # Seasonality effect
                seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * week_num / 52.0)
                # Festive Q4 bump
                if month in [10, 11]:
                    seasonality += 0.22
                # Harvest season bump for agriculture (Oct-Nov and Mar-Apr)
                if cat == 'Traditional Farming & Agri Produce' and month in [3, 4, 10, 11]:
                    seasonality += 0.30

                year_trend = (base_cfg['trend'] ** (year - 2019))
                base_units = base_cfg['units'] * reg_factor * seasonality * year_trend

                survey_shift_signal = survey_shifts.get(cat, 0.0)

                impact_score, risk_lvl, _ = calculate_disaster_impact_score(
                    severity=dis_sev,
                    duration_days=dis_dur,
                    geographic_spread=dis_geo,
                    population_affected=dis_pop,
                    economic_disruption=dis_econ,
                    supply_disruption=dis_supply,
                    industry=cat
                )

                for channel in channels:
                    noise = np.random.normal(0, 0.035)
                    is_offline = 'Offline' in channel

                    if region_hit:
                        if is_offline:
                            if 'Pandemic' in dis_type:
                                if cat == 'Restaurant Dine-in & Hospitality':
                                    channel_mult = 0.08
                                elif cat in ['Fresh Foods & Grocery', 'Medical Supplies & Healthcare']:
                                    channel_mult = 0.72
                                elif cat == 'Traditional Farming & Agri Produce':
                                    channel_mult = 0.40 # Mandi logistics blocked
                                else:
                                    channel_mult = 0.25
                            elif 'Drought' in dis_type or 'Hail' in dis_type:
                                # Traditional farming crisis
                                if cat == 'Traditional Farming & Agri Produce':
                                    channel_mult = 0.45 # Severe crop yield crash at Mandi
                                elif cat in ['Packaged Food & Beverages', 'Fresh Foods & Grocery']:
                                    channel_mult = 0.75 # Raw material shortages
                                else:
                                    channel_mult = 0.90
                            elif 'Modern AI Job Recession' in dis_type:
                                # Tech layoffs & white collar freeze
                                if cat in ['Luxury Goods & Premium Apparel', 'Restaurant Dine-in & Hospitality']:
                                    channel_mult = 0.55 # Sharp discretionary cut
                                elif cat == 'Electronics & Workstation Tech':
                                    channel_mult = 0.70 # Hardware purchase delay
                                else:
                                    channel_mult = 0.88
                            elif 'Flood' in dis_type or 'Cyclone' in dis_type:
                                channel_mult = max(0.18, 1.0 - (dis_sev * 0.085))
                            else: # Supply chain disruption
                                channel_mult = 0.80
                        else: # Online / AgriTech / D2C Channel
                            if 'Pandemic' in dis_type:
                                if cat == 'Medical Supplies & Healthcare':
                                    channel_mult = 2.45
                                elif cat in ['Fresh Foods & Grocery', 'Packaged Food & Beverages']:
                                    channel_mult = 1.85
                                elif cat == 'Electronics & Workstation Tech':
                                    channel_mult = 1.50
                                elif cat == 'Digital Services & AI Upskilling':
                                    channel_mult = 2.10
                                elif cat == 'Traditional Farming & Agri Produce':
                                    channel_mult = 1.65 # Direct AgriTech apps boom
                                elif cat == 'Restaurant Dine-in & Hospitality':
                                    channel_mult = 0.40 # Food delivery only
                                else:
                                    channel_mult = 1.15
                            elif 'Drought' in dis_type or 'Hail' in dis_type:
                                if cat == 'Traditional Farming & Agri Produce':
                                    channel_mult = 0.65 # Online AgriTech also feels supply crunch
                                else:
                                    channel_mult = 0.95
                            elif 'Modern AI Job Recession' in dis_type:
                                if cat == 'Digital Services & AI Upskilling':
                                    channel_mult = 2.30 # Massive surge in AI & Prompt Eng bootcamps / upskilling
                                elif cat in ['Luxury Goods & Premium Apparel']:
                                    channel_mult = 0.60 # E-commerce luxury cut
                                else:
                                    channel_mult = 0.92
                            elif 'Flood' in dis_type or 'Cyclone' in dis_type:
                                channel_mult = max(0.35, 1.0 - (dis_supply * 0.50))
                            else: # Supply chain
                                channel_mult = 0.85
                    else:
                        # Normal offline vs online split
                        channel_mult = 0.65 if is_offline else 0.35

                    price = base_cfg['price']
                    discount_pct = 0.05
                    if not region_hit and month in [10, 11]:
                        discount_pct = 0.18
                    elif region_hit and not is_offline:
                        discount_pct = 0.08

                    effective_units = max(5, int(base_units * channel_mult * (1.0 + noise)))
                    revenue = round(effective_units * price * (1.0 - discount_pct), 2)
                    normal_baseline = round(base_units * (0.65 if is_offline else 0.35) * price, 2)
                    sales_impact_pct = round(((revenue - normal_baseline) / normal_baseline) * 100.0, 2)

                    rows.append({
                        'date': week_str,
                        'year': year,
                        'month': month,
                        'quarter': quarter,
                        'week_of_year': week_num,
                        'region': region,
                        'category': cat,
                        'channel': channel,
                        'price': price,
                        'discount_pct': discount_pct,
                        'units_sold': effective_units,
                        'revenue_inr': revenue,
                        'baseline_revenue_inr': normal_baseline,
                        'sales_impact_pct': sales_impact_pct,
                        'is_disaster_active': 1 if region_hit else 0,
                        'disaster_name': dis_name,
                        'disaster_type': dis_type,
                        'disaster_severity': dis_sev,
                        'disaster_duration_days': dis_dur,
                        'supply_disruption_index': dis_supply,
                        'economic_disruption_index': dis_econ,
                        'geographic_spread_index': dis_geo,
                        'population_affected_index': dis_pop,
                        'mobility_reduction_pct': mobility_red,
                        'survey_channel_shift_pct': survey_shift_signal,
                        'industry_sensitivity': INDUSTRY_SENSITIVITY.get(cat, 1.0),
                        'disaster_impact_score': impact_score,
                        'risk_level': risk_lvl
                    })

    df_sales = pd.DataFrame(rows)
    df_sales.to_csv(output_sales_path, index=False)
    print(f"Generated Historical Sales Dataset: {len(df_sales)} records -> {output_sales_path}")

    df_sales.to_csv(output_processed_path, index=False)
    print(f"Saved Processed Dataset -> {output_processed_path}")

    return df_sales

if __name__ == "__main__":
    generate_aligned_sales_dataset()
