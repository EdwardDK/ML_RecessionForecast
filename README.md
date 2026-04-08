
# RecessionRiskForecast

**Real-time recession risk forecasting using machine learning**

---

## Overview

**RecessionRiskForecast** is a Python-based system that predicts the **probability of a U.S. recession in the next 6 months** using macroeconomic indicators from FRED and machine learning. The system employs **walk-forward cross-validation** and **XGBoost with calibrated probabilities** to provide robust, out-of-sample forecasts for rare economic events.

* **Precision-Recall AUC:** 0.1221 (primary metric)
* **ROC-AUC:** 0.5522
* **Forecast Horizon:** 6 months ahead
* **Input:** Monthly macroeconomic data from FRED
* **Output:** Probability of recession + visualizations

This project demonstrates **practical applications of time series ML**, feature engineering, and calibrated probability forecasting for financial decision-making.

---

## Features

* Fetch monthly economic series (unemployment, CPI, yields, consumer confidence, industrial production, recession indicator) from FRED
* Engineer lagged and derived features (yield curve, Sahm rule, CPI YoY, industrial production MoM)
* Train XGBoost classifier with **Optuna hyperparameter tuning** for each walk-forward fold
* Calibrate predicted probabilities using **isotonic regression** for rare event reliability
* Generate **walk-forward out-of-sample predictions** and evaluate using precision, recall, F1-score
* Visualizations:

  * Forecasted recession probability over time
  * Precision-Recall curve
  * Sahm Rule indicator chart

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/RecessionRiskForecast.git
cd RecessionRiskForecast
````

2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your FRED API key:

```
FRED_API_KEY=your_fred_api_key_here
```

**Key Dependencies:**

* Python 3.10+
* pandas, numpy, matplotlib
* FRED API (`fredapi`)
* scikit-learn (`sklearn`)
* xgboost (`xgboost`)
* optuna
* python-dotenv

---

## Usage

### 1. Fetch & Prepare Data

The script fetches U.S. macroeconomic series from FRED and resamples them monthly:

```bash
python main.py
```

Series fetched:

* Unemployment rate (`UNRATE`)
* CPI (`CPIAUCSL`)
* 10Y and 2Y Treasury yields (`DGS10`, `DGS2`)
* Consumer Confidence (`UMCSENT`)
* Industrial Production (`INDPRO`)
* Recession indicator (`USREC`)

---

### 2. Feature Engineering

Derived and lagged features include:

* Yield curve: 10Y − 2Y
* Sahm Rule: 3-month unemployment MA − 12-month unemployment min
* CPI year-over-year change
* Industrial production month-over-month change
* Lagged features for 1, 3, 6, 12 months
* Target: any recession within the next 6 months

---

### 3. Walk-Forward Cross-Validation

Evaluate the model using **walk-forward CV** to ensure robust, out-of-sample predictions:

* XGBoost classifier with Optuna hyperparameter tuning
* Calibrated probabilities (isotonic or sigmoid)
* Predictions threshold: 0.4 for rare events
* Outputs classification report with precision, recall, F1-score, PR-AUC, ROC-AUC

---

### 4. Final Model Training

Train the model on the **entire clean dataset** to generate current recession risk and top predictors.

---

### 5. Visualization

Generates a single chart `recession_forecast.png` that combines:

* Forecasted recession probability over time
* Precision-Recall curve
* Sahm Rule labor market stress indicator

```bash
python main.py
```

Example:

<img src="recession_forecast.png](https://github.com/EdwardDK/ML_RecessionForecast/blob/main/recession_forecast.png?raw=true" width="700" height="450">

---

## Results

* **Walk-Forward Classification Accuracy:** 65.8%
* **Precision-Recall AUC:** 0.1221 (primary metric for rare events)
* **ROC-AUC:** 0.5522
* **Current recession probability (next 6 months):** 3.8%
* **95% bootstrap CI:** [3.5%, 4.1%]

**Top predictors:**

```
yield_curve_lag12
sahm_rule_lag1
sahm_rule
indpro_mom
CPI
yield_curve_lag6
cpi_yoy_lag12
10Y
CONF_lag1
unrate_min_12
```

---

## Project Structure

```
RecessionRiskForecast/
├─ main.py                 # Main script: fetch, feature engineer, train, visualize
├─ .env                    # FRED API key
├─ recession_forecast.png  # Combined chart with probability, PR curve, Sahm rule
├─ README.md               # Project documentation
└─ requirements.txt        # Python dependencies
```

---

## Future Work

* Extend to **shorter horizon forecasts** (1–3 months)
* Add **macro sentiment or alternative data features**
* Explore **interactive dashboards** for financial analysts
* Test alternative ML models (LightGBM, TabNet) and ensemble approaches
