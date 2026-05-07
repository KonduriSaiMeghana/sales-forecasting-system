# Sales Forecasting System with FastAPI

## 📌 Project Overview

This project is an end-to-end Time Series Forecasting System built using Python and FastAPI.  
The system forecasts future sales using historical sales data and compares multiple forecasting algorithms to automatically select the best-performing model.

The project is designed with a production-style backend architecture including:
- data preprocessing
- feature engineering
- model training
- evaluation
- model persistence
- REST API deployment

---

# 🚀 Features

✅ Time Series Forecasting  
✅ Multiple Model Training & Comparison  
✅ Automatic Best Model Selection  
✅ Feature Engineering Pipeline  
✅ REST API using FastAPI  
✅ Swagger API Documentation  
✅ Production-Style Project Structure  
✅ Visualization & EDA Notebooks

---

# 📂 Project Structure

```text
forecasting-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── saved_models/
│   └── model.pkl
│
├── src/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   └── model_selection.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   │
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── load_data.py
│   │   ├── missing_dates.py
│   │   └── train_test_split.py
│   │
│   └── utils/
│       ├── helpers.py
│       ├── logger.py
│       └── metrics.py
│
├── app.py
├── train_pipeline.py
├── requirements.txt
└── README.md
```

# 📊 Dataset

The dataset contains historical beverage sales data across multiple states in the United States.  
It is used to build a time-series forecasting system capable of predicting future sales trends.

The dataset captures:
- state-wise sales performance
- temporal trends
- seasonality patterns
- category-based sales information

---

# 📁 Dataset Information

| Attribute | Details |
|---|---|
| Dataset Type | Time Series Sales Data |
| Domain | Retail / Beverage Sales |
| Total Records | 8084 |
| Total Features | 4 |
| Time Period | 2019 - 2020 |
| Granularity | Daily Sales |
| Categories | Beverage Products |

---

# 🧾 Dataset Columns

| Column | Data Type | Description |
|---|---|---|
| `state` | Object | Name of the US state |
| `date` | Datetime | Sales transaction date |
| `sales` | Float | Total sales amount |
| `category` | Object | Product category |

---

# 📌 Sample Dataset

| state | date | sales | category |
|---|---|---|---|
| Alabama | 2019-01-12 | 109574036.0 | Beverages |
| Arizona | 2019-01-12 | 109101594.6 | Beverages |
| Arkansas | 2019-01-12 | 58049432.2 | Beverages |
| California | 2019-01-12 | 444766890.6 | Beverages |
| Colorado | 2019-01-12 | 89816716.3 | Beverages |

---

# 📈 Dataset Statistics

| Metric | Value |
|---|---|
| Total Rows | 8084 |
| Total Columns | 4 |
| Average Sales | 165,858,000 |
| Maximum Sales | 985,374,600 |
| Minimum Sales | 9,732,839 |
| Missing Values | 0 |

---

# 🔍 Exploratory Data Analysis (EDA)

Several exploratory data analysis techniques were performed to better understand the dataset:

## Visualizations Included

- Total Sales Trend Over Time
- Monthly Sales Trend
- Top States by Sales
- Category-wise Sales Analysis
- Rolling Mean Trend
- Missing Values Heatmap
- Feature Correlation Heatmap

---

# 📊 Key Insights

- Sales show strong upward and seasonal trends over time.
- Certain states contribute significantly higher sales volumes.
- Rolling mean analysis indicates stable long-term growth.
- Feature correlation analysis shows lag variables strongly influence future sales.
- Holiday periods slightly impact sales behavior.

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed before model training:

## ✅ Data Cleaning
- Standardized column names
- Converted date column to datetime format
- Sorted data chronologically

## ✅ Missing Value Handling
- Filled missing sales values
- Handled missing dates using interpolation

## ✅ Time Series Processing
- Maintained chronological order
- Used time-based train-test split to prevent data leakage

---

# 🛠️ Feature Engineering

Several time-series features were engineered to improve forecasting performance.

## Lag Features
These features capture previous sales behavior.

- `lag_1`
- `lag_7`
- `lag_30`

## Rolling Statistics
These features capture trend and volatility.

- `rolling_mean_7`
- `rolling_std_7`

## Calendar Features
These features capture seasonality patterns.

- `day_of_week`
- `month`

## Holiday Features
- `is_holiday`

---

