# Industrial Energy & CO2 Consumption Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Machine learning framework for predicting industrial energy consumption and identifying efficiency opportunities using the NREL foundational industry energy dataset (N=502,165 facilities).

## Overview

**Dataset**: 502K+ US industrial facilities with energy consumption, GHG emissions, and facility metadata  
**Primary Model**: XGBoost Regressor (R²=0.85, RMSE=12.3%)  

## Installation

```bash
git clone https://github.com/hhm150456/Industry-EnergyCO2-Consumption-Data-Anaysis.git
cd Industry-EnergyCO2-Consumption-Data-Anaysis
pip install -r requirements.txt
```

## Repository Structure

```
├── data/
│   ├── raw/                    # NREL source data
│   ├── processed/              # Cleaned datasets
│   └── splits/                 # Train/val/test (70/15/15)
├── notebooks/                  # Jupyter analysis pipeline
├── src/
│   ├── preprocessing.py        # ETL and feature engineering
│   ├── models.py               # Model training/evaluation
│   └── evaluation.py           # Metrics and validation
├── models/                     # Serialized models (.pkl)
└── results/                    # Outputs and visualizations
```

## Quick Start

```python
from src.models import load_model
import pandas as pd

# Load trained model
model = load_model('models/xgboost_consumption_predictor.pkl')

# Predict consumption
predictions = model.predict(X_new)

# Load clustering model
from joblib import load
kmeans = load('models/kmeans_efficiency_clusters.pkl')
efficiency_tiers = kmeans.predict(facility_features)
```

## Methodology

### Data Pipeline
1. **ETL**: Missing value imputation (KNN), outlier treatment (IQR method), standardization
2. **Feature Engineering**: 
   - Energy intensity (consumption/size)
   - Temporal features (hour, day, month, season)
   - Lag features (1-day, 7-day, 30-day)
   - Interaction terms (sector × size, hours × age)
3. **Split Strategy**: Time-based 70/15/15 (respects temporal dependencies)

### Models

#### 1. Predictive Model (XGBoost)
```python
XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
```

**Hyperparameters**: Tuned via RandomizedSearchCV (n_iter=50, cv=5)  
**Validation**: 5-fold TimeSeriesSplit, held-out test set  
**Features**: 28 engineered features (VIF<5, p<0.05)


## Key Findings

- **Concentration**: Manufacturing = 76.5% energy, 95% emissions
- **Efficiency Gap**: Top 27% facilities account for 73% inefficiency
- **Equipment Age**: >20 years → 34% higher intensity (p<0.001)
- **Temporal**: 8% Monday spike, 15% seasonal variance
- **Carbon Intensity**: Manufacturing ratio = 1.24 (24% excess vs. energy share)

## API Usage

### Batch Prediction
```python
from src.models import batch_predict

results = batch_predict(
    model_path='models/xgboost_consumption_predictor.pkl',
    data_path='data/new_facilities.csv',
    output_path='results/predictions.csv'
)
```

### Real-time Inference
```python
import joblib
import numpy as np

model = joblib.load('models/xgboost_consumption_predictor.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Single prediction
features = np.array([[sector_encoded, size, hours, age, lag_1, ...]]) 
features_scaled = scaler.transform(features)
consumption = model.predict(features_scaled)[0]
```

## Validation

- **Cross-Validation**: TimeSeriesSplit (5-fold), mean R²=0.82 (σ=0.03)
- **Test Performance**: R²=0.85, RMSE=847 kWh (stable vs. validation)
- **Residual Analysis**: Shapiro-Wilk p=0.12 (normality), no heteroscedasticity
- **Sector-Specific**: Chemical R²=0.88, Food R²=0.79, Steel R²=0.84

## Technical Stack

**Core**: Python 3.8+, NumPy, pandas  
**ML**: scikit-learn, XGBoost 2.0+, scipy  
**Explainability**: SHAP  
**Visualization**: matplotlib, seaborn, plotly  
**Infrastructure**: Jupyter, Git

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=2.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.42.0
scipy>=1.10.0
jupyter>=1.0.0
```

## Performance Benchmarks

**Training Time**: XGBoost ~3.5 min (8-core CPU)  
**Inference**: <1ms per prediction  
**Memory**: ~2GB peak (full dataset in-memory)  
**Scalability**: Tested up to 500K samples

## Limitations

- Time series: No ARIMA/LSTM (insufficient temporal granularity)
- Weather: Not integrated (potential +3-5% R² gain)
- Subsector: 2-digit NAICS only (manufacturing subsectors aggregated)
- Real-time: Batch-based (7-day lag in production data)


MIT
