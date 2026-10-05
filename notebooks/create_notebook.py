import json
import os

os.makedirs('notebooks', exist_ok=True)
notebook_path = 'notebooks/01_disaster_aware_sales_forecasting_pipeline.ipynb'

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# 🛡️ Disaster-Aware Sales Forecasting & Impact Analysis System\n',
            '### End-to-End Experimental Pipeline & Model Validation\n',
            'This notebook covers:\n',
            '1. Loading Multi-Year Sales, Disaster Registry & BCG India COVID-19 Survey (N=2,106)\n',
            '2. Feature Engineering (Lags, Rolling Averages, Cyclical Calendar, Disaster Interactions)\n',
            '3. Disaster Impact Score (0-100) Formulation\n',
            '4. Strict Time-Based Validation (2019-2022 Train, 2023 Out-of-Time Test)\n',
            '5. Model Benchmarking: Linear Regression, Random Forest, LightGBM, XGBoost\n',
            '6. SHAP Explainability & Scenario Simulation (Traditional Farming & AI Recession)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'import sys\n',
            'sys.path.insert(0, os.path.abspath(".."))\n',
            'import numpy as np\n',
            'import pandas as pd\n',
            'import matplotlib.pyplot as plt\n',
            'import seaborn as sns\n',
            '\n',
            '# Load raw datasets\n',
            'sales_df = pd.read_csv("../data/raw/historical_sales.csv")\n',
            'disasters_df = pd.read_csv("../data/raw/disaster_events.csv")\n',
            'survey_df = pd.read_csv("../data/raw/survey_bcg_covid19.csv")\n',
            '\n',
            'print(f"Sales records: {len(sales_df)}")\n',
            'print(f"Disaster events: {len(disasters_df)}")\n',
            'print(f"Survey responses: {len(survey_df)}")\n',
            'sales_df.head()'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Feature Engineering & Preprocessing\n',
            'from src.preprocessing import prepare_and_preprocess_data\n',
            '\n',
            'X_train, y_train, X_test, y_test, encoders, feature_cols = prepare_and_preprocess_data(\n',
            '    data_path="../data/processed/final_dataset.csv",\n',
            '    split_year=2023,\n',
            '    models_dir="../models"\n',
            ')\n',
            '\n',
            'print(f"Training samples (2019-2022): {len(X_train)}")\n',
            'print(f"Test samples (2023): {len(X_test)}")\n',
            'print(f"Total features engineered: {len(feature_cols)}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Load Model Benchmark Metrics\n',
            'import json\n',
            'with open("../models/evaluation_metrics.json", "r") as f:\n',
            '    metrics = json.load(f)\n',
            '\n',
            'benchmark_df = pd.DataFrame(metrics["metrics"]).T\n',
            'print(f"Best Model: {metrics[\'best_model\']}")\n',
            'benchmark_df[["mae", "rmse", "mape_pct", "r2_score"]]'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Run What-If Simulation Test\n',
            'from src.predict import predict_scenario\n',
            '\n',
            'result = predict_scenario(\n',
            '    region="Maharashtra",\n',
            '    category="Traditional Farming & Agri Produce",\n',
            '    channel="Offline Store / Mandi",\n',
            '    disaster_type="Traditional Farming Drought & Crop Shock",\n',
            '    disaster_severity=8.0,\n',
            '    disaster_duration_days=45,\n',
            '    supply_disruption_index=0.85,\n',
            '    economic_disruption_index=0.60,\n',
            '    geographic_spread_index=0.50,\n',
            '    population_affected_index=0.45,\n',
            '    mobility_reduction_pct=15.0,\n',
            '    baseline_revenue=500000.0,\n',
            '    models_dir="../models"\n',
            ')\n',
            '\n',
            'print("Simulation Summary:")\n',
            'print(f"Predicted Sales: INR {result[\'predicted_revenue\']:,.2f}")\n',
            'print(f"Impact %: {result[\'impact_percentage\']}%")\n',
            'print(f"Impact Score: {result[\'impact_score\']} / 100 ({result[\'risk_level\']} Risk)")\n',
            'print(f"Estimated Recovery: {result[\'estimated_recovery_weeks\']} weeks")'
        ]
    }
]

nb = {
    'cells': cells,
    'metadata': {
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print('Created Jupyter Notebook at', notebook_path)
