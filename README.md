# ⚡ Disaster-Aware Sales Forecasting & Impact Analysis System

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, enterprise-grade decision support platform built with **Python, Machine Learning, and Streamlit** for quantified demand forecasting and shock resilience analysis across natural disasters, climate crop failures, pandemics, and macroeconomic recessions.

Designed and optimized for a **BE AI&DS Final Year Capstone Project** and 1-click deployment on **GitHub + Streamlit Cloud**.

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Project Architecture](#-project-architecture)
3. [Streamlit UI Navigation](#-streamlit-ui-navigation)
4. [Dataset & Schema Assumptions](#-dataset--schema-assumptions)
5. [Automated Data Preprocessing Engine](#-automated-data-preprocessing-engine)
6. [Machine Learning & Zero-Leakage Validation](#-machine-learning--zero-leakage-validation)
7. [Disaster Impact Score (0–100) Formulation](#-disaster-impact-score-0100-formulation)
8. [Local Installation & Setup](#-local-installation--setup)
9. [Deployment to Streamlit Cloud](#-deployment-to-streamlit-cloud)
10. [MVP Feature Verification Checklist](#-mvp-feature-verification-checklist)

---

## 🌟 Executive Summary

Traditional retail and agricultural sales forecasting systems rely solely on historical autoregressive patterns and seasonal trends, leading to severe supply-chain collapse during exogenous disruptions (e.g., COVID-19 lockdowns, Kharif drought shocks, cyclone logistics bottlenecks, or AI layoff recessions).

This system bridges that gap by integrating:
- **Exogenous Disruption Signals** into tabular machine learning models.
- **Before → During → After Disruption Phase Analysis** tracking 95% baseline recovery crossing.
- **Dynamic What-If Scenario Stress Testing** allowing business leaders to simulate custom supply and demand shocks.
- **Defensible 0–100 Disaster Impact Score** categorizing risk posture (Low, Medium, High, Critical).

---

## 🏛️ Project Architecture

```text
disaster-sales-forecasting/
│
├── app.py                      # Core Streamlit multi-section application
├── requirements.txt            # Streamlit Cloud-compatible dependencies
├── README.md                   # Comprehensive technical documentation
│
├── data/
│   ├── sample_sales.csv        # Built-in multi-year retail & agri sales dataset
│   └── sample_disasters.csv    # Built-in historical disruption event registry
│
├── models/
│   ├── model.pkl               # Primary trained ML model (XGBoost / Random Forest)
│   ├── best_model.pkl          # Top-performing benchmark model
│   ├── evaluation_metrics.json # Out-of-time evaluation metrics (MAE, RMSE, MAPE, R²)
│   └── feature_columns.pkl     # Persisted feature column schema
│
├── src/
│   ├── preprocessing.py        # Schema validation, missing value imputation, deduplication
│   ├── features.py             # Lags, rolling averages, and interaction feature engineering
│   ├── impact_score.py         # Composite 0–100 Disaster Impact Score algorithm
│   ├── forecasting.py          # ML horizon prediction & What-If scenario forecasting
│   ├── recovery.py             # Phase tracking & 95% baseline recovery crossing
│   └── explain.py              # Feature contribution diagnostics
│
└── notebooks/
    └── model_training.ipynb    # Jupyter experiment and training pipeline
```

---

## 🧭 Streamlit UI Navigation

The application is organized into **five logical, intuitive tabs**:

| Section | Key Features |
| :--- | :--- |
| **1. 📊 Dashboard** | Executive KPI cards (Total Sales, Average Sales, Growth %, Disaster Impact %, Risk Level), historical sales trend with shaded disaster overlays, and sector breakdown charts. |
| **2. 📁 Upload Data** | Upload custom sales and disaster CSVs, automatic alias mapping, automated missing value imputation, deduplication, date validation, quality reports, and sample template downloads. |
| **3. 🚨 Disaster Impact Analysis** | Interactive shock parameters (type, severity, duration, region), Before → During → After sales comparison, 0–100 Disaster Impact Score, risk categorization, and CSV export. |
| **4. 🔮 Sales Forecast** | User-selected forecast horizon (1 to 12 weeks), model choice (Random Forest, XGBoost, Linear Regression), confidence intervals, out-of-time test metrics (**MAE, RMSE, MAPE, R²**), and CSV export. |
| **5. 🎯 What-If Simulation** | Stress-test sales under synthetic supply-chain bottlenecks and consumer shift percentages, compare normal vs disaster scenarios, view recovery timelines, and export results. |

---

## 📋 Dataset & Schema Assumptions

### 1. Sales Dataset (`data/sample_sales.csv`)
The system accepts any CSV with flexible column naming. The preprocessing layer automatically maps the following aliases:

| Logical Column | Description | Accepted Aliases | Fallback if Missing |
| :--- | :--- | :--- | :--- |
| `date` | Timestamp / Order Date | `date`, `Date`, `order_date`, `timestamp`, `sales_date` | **Mandatory** |
| `revenue_inr` | Sales / Revenue | `revenue_inr`, `revenue`, `sales`, `Sales`, `total_sales`, `amount` | **Mandatory** |
| `units_sold` | Quantity sold | `units_sold`, `units`, `quantity`, `qty`, `volume` | Generated from revenue |
| `region` | Geographic location | `region`, `Region`, `state`, `zone`, `market`, `location` | `"National Market"` |
| `category` | Product sector | `category`, `Category`, `sector`, `industry`, `item_category` | `"General Retail & FMCG"` |
| `channel` | Fulfillment channel | `channel`, `Channel`, `fulfillment_channel`, `platform` | `"Offline Store / Mandi"` |

### 2. Disaster Dataset (`data/sample_disasters.csv`) - *Optional*

| Column | Type | Description |
| :--- | :--- | :--- |
| `disaster_name` | String | Name of event (e.g., Cyclone Fani, COVID-19 Lockdown) |
| `disaster_type` | String | Category (Cyclone, Flood, Pandemic, Drought, Recession) |
| `start_date` | Date | Start date (`YYYY-MM-DD`) |
| `end_date` | Date | End date (`YYYY-MM-DD`) |
| `severity` | Float (1-10) | Shock intensity |
| `affected_regions` | String | Comma-separated regions or `"All India"` |

---

## ⚙️ Automated Data Preprocessing Engine

When a CSV is uploaded or loaded from defaults, `src/preprocessing.py` executes:
1. **Schema Validation & Alias Matching:** Inspects headers case-insensitively and maps recognized aliases.
2. **Date Alignment & Error Coercion:** Parses strings into datetime objects; safely drops invalid timestamp rows.
3. **Missing Value Imputation:** Imputes missing revenue with regional/category median; fills missing categories with sensible defaults.
4. **Deduplication:** Identifies and drops duplicate `(date, region, category, channel)` rows.
5. **Calendar Feature Extraction:** Generates `year`, `month`, `quarter`, and `week_of_year`.
6. **Disaster Alignment:** Intersects sales timeline with disaster active windows and flags `is_disaster_active`.

---

## 🤖 Machine Learning & Zero-Leakage Validation

To prevent **temporal data leakage**, the system employs strict **out-of-time splitting**:
- **Training Set:** 2019 – 2022 (Historical normal & disruption cycles)
- **Evaluation Test Set:** 2023 (Held-out out-of-time benchmark)

### Evaluated Model Benchmarks:
| Model Architecture | MAE (₹) | RMSE (₹) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **₹123,505.42** | **₹418,946.60** | **6.65%** | **0.9425** |
| **Random Forest Regressor** | ₹135,725.99 | ₹514,948.80 | 6.84% | 0.9131 |
| **Linear Regression (Baseline)** | ₹340,687.80 | ₹643,497.02 | 37.14% | 0.8644 |

---

## 🧮 Disaster Impact Score (0–100) Formulation

The composite **Disaster Impact Score** is calculated via a normalized weighted index:

$$\text{Base Score} = \left( 0.25 \cdot S + 0.25 \cdot SC + 0.15 \cdot D + 0.15 \cdot E + 0.10 \cdot G + 0.10 \cdot P \right) \times 100$$

$$\text{Final Impact Score} = \min\left(100, \text{Base Score} \times \text{Industry Sensitivity}\right)$$

Where:
- $S$: Normalized Severity ($0 - 1$)
- $SC$: Supply Disruption Index ($0 - 1$)
- $D$: Normalized Duration ($0 - 1$, bounded at 120 days)
- $E$: Macro Economic Shock Index ($0 - 1$)
- $G$: Geographic Spread ($0 - 1$)
- $P$: Population Affected ($0 - 1$)

### Risk Categorization:
- **0 – 25:** 🟢 **Low Risk**
- **26 – 50:** 🟡 **Medium Risk**
- **51 – 75:** 🟠 **High Risk**
- **76 – 100:** 🔴 **Critical Risk**

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Steps:
```bash
# 1. Clone the repository
git clone https://github.com/AnushkaJagtapp/sales-forecasting.git
cd sales-forecasting

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

The app will be available locally at `http://localhost:8501`.

---

## 🚀 Deployment to Streamlit Cloud

This project is configured for **Streamlit Cloud**:

1. Push the code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Deploy disaster-aware sales forecasting MVP"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository: `AnushkaJagtapp/sales-forecasting`.
5. Set the branch: `main`.
6. Set the Main file path: `app.py`.
7. Click **"Deploy!"**.

---

## ✅ MVP Feature Verification Checklist

- [x] **1. CSV Sales Data Upload:** Upload custom sales CSV with clear column validation and automatic fallback to built-in dataset.
- [x] **2. Optional Disaster Data Upload:** Upload disaster CSV with graceful error handling and default sample registry.
- [x] **3. Automatic Data Preprocessing:** Imputes missing values, removes duplicates, validates dates, and logs actions.
- [x] **4. Dashboard / EDA:** 5 KPI cards (Total Sales, Avg Sales, Growth, Impact %, Risk), historical trends, category & regional charts.
- [x] **5. Disaster Impact Analysis:** User controls for type/severity/duration/region, Before → During → After trajectory, 0-100 Impact Score, and risk classification.
- [x] **6. ML Sales Forecasting:** 1-12 week user-selectable horizon, Random Forest & XGBoost, table + line chart, and **MAE, RMSE, MAPE, R²** evaluation metrics.
- [x] **7. What-If Disaster Simulation:** Modifiable severity, duration, supply disruption %, and consumer shift % comparing normal vs disaster scenarios with recovery timeline.
- [x] **8. Download Results:** 1-click CSV download buttons for forecasts, impact reports, What-If simulations, and sample data templates.
- [x] **9. Streamlit Structure:** Organized into the 5 requested tabs.
- [x] **10. Deployment Constraints Met:** Pure Streamlit Cloud + GitHub deployment without external DBs, Docker, or unnecessary cloud dependencies.
