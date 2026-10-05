# 🛡️ Disaster-Aware Sales Forecasting and Impact Analysis System

> An enterprise-grade AI decision support system integrating historical sales time series, multi-category disruption event dynamics (Pandemics, Floods, Cyclones, Traditional Farming Shocks, Modern AI Job Recessions), consumer sentiment survey signals (BCG India Wave 1, N=2,106), 0–100 composite impact scoring, and SHAP explainability.

---

## 📌 Executive Summary

Traditional sales forecasting models assume future consumer demand roughly mirrors historical patterns. However, sudden disruptive events—such as natural disasters, pandemics, crop failures, or white-collar AI layoffs—invalidate historical trends by causing asymmetric channel shifts (Offline Mandi/Store $\to$ Online/D2C), supply bottlenecks, and sector-specific demand shocks.

This system answers three critical enterprise questions:
1. **Forecast**: *"Given previous sales and active disruption shocks, how much will we sell next?"*
2. **Impact & Risk**: *"How severely are operations impacted, and what is our enterprise risk tier (Low, Moderate, High, Critical)?"*
3. **Recovery & Strategy**: *"How long will it take to recover 95% baseline sales, and what what-if actions minimize loss?"*

---

## 🏗️ 6-Layer System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. MULTI-SOURCE DATA INGESTION                              │
│  • Multi-Year Historical Sales (2019–2023, 6 Regions)       │
│  • Real-Calibrated Disaster & Disruption Event Registry     │
│  • BCG India COVID-19 Consumer Sentiment Survey (N=2,106)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING & STRICT TIME-BASED VALIDATION             │
│  • 0% Data Leakage Split: Train (2019–2022) | Test (2023)  │
│  • Missing Value Handling & Categorical Label Encoding      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING & INTERACTION DOMAIN                 │
│  • Time Lags (1, 2, 4, 8 Weeks) & Rolling Window Statistics │
│  • Cyclical Calendar Encodings (Sine/Cosine Week & Month)   │
│  • Interactions: (Severity × Sensitivity), (Supply × Channel)│
│  • Composite 0–100 Disaster Impact Score Computation        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MACHINE LEARNING FORECASTING ENGINE                      │
│  • Linear Regression (Baseline)                             │
│  • Random Forest Regressor                                  │
│  • LightGBM Regressor                                       │
│  • XGBoost Regressor (Best Model, R² = 0.9430, MAPE = 6.64%)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DECISION SUPPORT & RECOVERY ENGINE                       │
│  • Net Sales Impact % Calculation                           │
│  • Before-During-After Disruption Phase Segmentation        │
│  • 95% Baseline Sales Recovery Horizon Estimation           │
│  • Real-time What-If Simulation (Normal vs Mod vs Severe)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. EXPLAINABILITY & STREAMLIT ENTERPRISE UI                 │
│  • SHAP Waterfall Feature Attributions (INR Impact)         │
│  • Automated Natural-Language Executive Narratives          │
│  • 7 Interactive Dashboard Tabs & Viva Examination Guide    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧮 Mathematical Formulation

### 1. Composite Disaster Impact Score ($0 - 100$)
$$ImpactScore = \min\left(100, \, 100 \times \left( \sum_{i=1}^{6} w_i \cdot x_{i,\text{norm}} \right) \times S_{\text{industry}} \right)$$

| Component | Variable ($x_i$) | Weight ($w_i$) | Normalization | Operational Justification |
| :--- | :--- | :---: | :---: | :--- |
| **Severity** | $S$ | **0.25** | $\frac{S}{10}$ | Direct shock intensity & damage scale |
| **Supply Disruption** | $SC$ | **0.25** | Bounded $[0, 1]$ | Inventory deficit & logistics paralysis |
| **Duration** | $D$ | **0.15** | $\min(1.0, \frac{D}{120})$ | Temporal persistence of the event |
| **Economic Disruption**| $E$ | **0.15** | Bounded $[0, 1]$ | Macro demand & purchasing power shock |
| **Geographic Spread** | $G$ | **0.10** | Bounded $[0, 1]$ | Multi-state vs localized exposure |
| **Population Affected**| $P$ | **0.10** | Bounded $[0, 1]$ | Consumer & workforce density impacted |

$$\sum w_i = 0.25 + 0.25 + 0.15 + 0.15 + 0.10 + 0.10 = 1.00$$

### 2. Risk Level Categorization
* **$0 - 25$**: 🟢 **Low Risk** — Minor friction, standard buffer stock absorption.
* **$26 - 50$**: 🟡 **Moderate Risk** — Localized supply delays, moderate channel shift.
* **$51 - 75$**: 🔴 **High Risk** — Severe supply breakdown, substantial offline revenue loss.
* **$76 - 100$**: 🟣 **Critical Risk** — Systemic shutdown / multi-month disruption, urgent D2C pivot required.

### 3. Disruption Impact Percentage
$$Impact\% = \frac{\text{During Sales} - \text{Baseline Sales}}{\text{Baseline Sales}} \times 100$$

### 4. Recovery Horizon Detection
$$\text{Recovery Week} = \min \left\{ t > t_{\text{end}} \mid \text{Sales}_t \ge 0.95 \times \text{Baseline Sales} \right\}$$

---

## 📊 Measured Model Performance (Out-of-Time Test Set: 2023)

All models were evaluated strictly on out-of-time unseen 2023 sales records (5,616 samples) after training on 2019–2022 (22,464 samples):

| Model Name | MAE (INR) | RMSE (INR) | MAPE (%) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **₹123,341.11** | **₹417,113.61** | **6.64%** | **0.9430** | 🏆 **Best Model** |
| **Random Forest Regressor** | ₹128,948.76 | ₹481,307.27 | 6.70% | 0.9241 | Strong Baseline |
| **Linear Regression (Baseline)**| ₹340,687.80 | ₹643,497.02 | 37.14% | 0.8644 | Linear Reference |
| **LightGBM Regressor** | ₹178,389.87 | ₹714,278.93 | 7.40% | 0.8329 | Fast Leaf-wise |

---

## 🌾 Traditional Farming & 💻 Modern AI Job Recession Specializations

### 1. Traditional Farming Disruption (Monsoon Shocks & Mandi Bottlenecks)
* **Vulnerability**: APMC Mandis experience $-40\%$ to $-65\%$ collapse in crop arrivals during monsoon deficit or hailstorms.
* **Price vs Volume Paradox**: Farmgate physical volumes crash, causing wholesale inflation while farmers face yield collapse.
* **AgriTech Pivot**: Direct farm-to-consumer and AgriTech platforms surge ($+42\%$ to $+65\%$), bypassing paralyzed physical transport.

### 2. Modern AI Job Recession & Tech Layoff Shocks
* **White-Collar Freeze**: Tech hub layoffs in Bengaluru, Pune, NCR trigger sharp contraction in Luxury Apparel ($-35\%$) and Restaurant Dine-in ($-45\%$).
* **Upskilling Surge**: Direct surge in AI Certification, Prompt Engineering, and Cloud upskilling demand ($+130\%$).

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
* Python 3.10+ installed
* Anaconda / Pip package manager

### 2. Clone / Open Workspace
```bash
cd "d:/sales forecasting"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Regenerate Datasets & Retrain Models
```bash
python data_generator.py
python src/train.py
```

### 5. Launch the Streamlit Enterprise Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎓 Viva Voce Preparation Guide (Questions & Defensible Answers)

### Q1: Why is this not just another basic regression project?
> **Answer**: Standard regression models univariate historical sales under the assumption that past patterns persist. Our system builds a **multivariate disaster-aware decision architecture** integrating external disruption severity, duration, supply bottlenecks, regional sensitivities, survey behavioral signals, and 0–100 composite impact scores to model asymmetric demand and channel reallocations.

### Q2: How did you prevent Data Leakage?
> **Answer**: We strictly enforced **Time-Based Splitting** (Training on 2019–2022, Testing on unseen 2023). We never use random k-fold shuffling. All lag features and rolling aggregations use `shift(1)` to ensure information from the forecast period is never accessible during feature calculation.

### Q3: How is the 0–100 Disaster Impact Score justified?
> **Answer**: The Disaster Impact Score is a **transparent, project-defined composite index**. We normalize operational disruption factors (Severity 25%, Supply Disruption 25%, Duration 15%, Economic Disruption 15%, Geographic Spread 10%, Population 10%) summing to 1.00, modulated by industry sensitivity. We defend it as an engineered decision feature, not an international universal standard.

### Q4: What is the relationship between Survey Data and Sales Data?
> **Answer**: Survey data (BCG Wave 1, N=2,106) captures **stated consumer intentions/sentiment** (e.g. 55% shifting to online electronics). Sales data captures **actual realized transactions**. The survey is converted into an external behavioral feature (`survey_channel_shift_pct`) feeding the ML forecasting model.

---

## 📁 Repository Structure

```text
d:/sales forecasting/
├── data/
│   ├── raw/
│   │   ├── historical_sales.csv       # Multi-year sales records (28,080 rows)
│   │   ├── disaster_events.csv        # Disruption event registry (10 events)
│   │   └── survey_bcg_covid19.csv     # BCG Wave 1 survey responses (N=2,106)
│   └── processed/
│       ├── final_dataset.csv          # Aligned time-series dataset
│       └── featured_dataset.csv       # Preprocessed feature matrix
├── models/
│   ├── best_model.pkl                 # Best performing XGBoost model
│   ├── xgboost.pkl                    # Serialized XGBoost model
│   ├── random_forest.pkl              # Serialized Random Forest model
│   ├── lightgbm.pkl                   # Serialized LightGBM model
│   ├── linear_regression_baseline.pkl # Serialized Linear Regression model
│   ├── feature_columns.pkl            # 48 feature names list
│   └── evaluation_metrics.json        # Out-of-time benchmark metrics
├── src/
│   ├── impact_score.py                # 0-100 Impact Score formula & weights
│   ├── features.py                    # Lags, rolling stats, interactions
│   ├── preprocessing.py               # Time-based splitting & encoding
│   ├── recovery.py                    # Phase analysis & 95% recovery threshold
│   ├── explain.py                     # SHAP attribution & business narratives
│   ├── train.py                       # Out-of-time training pipeline
│   └── predict.py                     # Inference & scenario simulation
├── app.py                             # 7-Tab Streamlit enterprise dashboard
├── data_generator.py                  # Realistic aligned dataset generator
├── requirements.txt                   # Dependency manifest
└── README.md                          # Full documentation & viva guide
```
