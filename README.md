# 📊 Multi-Store Retail Demand Forecaster

An end-to-end Machine Learning project that forecasts daily sales across multiple retail stores using LightGBM. The project includes an interactive Streamlit dashboard for generating forecasts, comparing stores, monitoring demand patterns, and analyzing model performance.

## 🚀 Project Overview

Retail businesses need accurate sales forecasts to manage inventory, staffing, and promotional activities.

This project uses historical retail sales data to predict future sales for individual stores and provides an interactive dashboard to explore the forecasts.

**Key highlights:**

* Multi-store daily sales forecasting.
* LightGBM-based forecasting model.
* Interactive Streamlit dashboard.
* Editable future promotion and store-opening calendar.
* Demand alerts based on historical weekday sales.
* Store-level comparison and model performance analysis.

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* LightGBM
* Scikit-learn
* Plotly
* Streamlit
* Joblib
* PyArrow

## ✨ Features

### 1. Sales Forecasting

Generate future daily sales predictions for selected stores with a configurable forecasting horizon.

### 2. Interactive Dashboard

Explore forecasts through an interactive Streamlit interface.

### 3. Future Calendar

Adjust future store-opening and promotion assumptions.

### 4. Demand Alerts

Identify unusually high or low predicted demand using historical weekday sales patterns.

### 5. Store Comparison

Compare sales patterns and forecasts across different stores.

### 6. Model Performance

Explore backtesting metrics and horizon-wise forecasting errors.

## 🧠 Machine Learning Model

**Final Model:** LightGBM

The forecasting pipeline uses historical sales information and engineered features to generate predictions.

The exported model and supporting data files are stored locally in the project.

## 📁 Project Structure

```text
RetailForecaster/
│
├── dashboard/
│   ├── app.py
│   └── charts.py
│
├── data/
│   ├── history.parquet
│   ├── store_meta.parquet
│   └── store_summary.parquet
│
├── models/
│   ├── lgbm_final.joblib
│   ├── final_metrics.json
│   └── model_card.json
│
├── scripts/
│   ├── check_app_logic.py
│   └── make_requirements.py
│
├── src/
│   ├── features.py
│   ├── forecasting.py
│   ├── backtest.py
│   ├── evaluation.py
│   └── insights.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Piyush700gitt/multi-store-retail-demand-forecaster.git
```

### 2. Navigate to the project directory

```bash
cd multi-store-retail-demand-forecaster
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## 📈 Model Evaluation

The project evaluates forecasting performance using backtesting and error metrics.

The dashboard provides:

* Model evaluation metrics.
* Horizon-wise forecasting errors.
* Store-level comparisons.

See `models/final_metrics.json` for the recorded evaluation results.

## ⚠️ Limitations

* Forecasts depend on historical patterns and the supplied future assumptions.
* Unexpected market changes and external events may affect actual sales.
* Predictions are estimates, not guaranteed sales outcomes.
* Forecast dates are limited by the historical dataset and configured forecasting horizon.

## 👨‍💻 Author

**Piyush Shrivastava**

GitHub: [Piyush700gitt](https://github.com/Piyush700gitt)

---

⭐ If you find this project interesting, consider giving the repository a star!
