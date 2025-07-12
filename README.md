# 🛒 Walmart Sales Analysis & Forecast Dashboard

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A full machine learning pipeline and interactive dashboard to analyze and forecast Walmart sales. Combines model experimentation with an intuitive Streamlit app for exploring weekly sales trends and predictions.

---

## 📋 Table of Contents

1. [📌 Project Overview](#-project-overview)
2. [📊 Dataset](#-dataset)
3. [🔄 Pipeline & Workflow](#-pipeline--workflow)
4. [📈 Insights](#-insights)
5. [💻 Usage Examples](#-usage-examples)
6. [🤝 Contributing](#-contributing)
7. [📬 Contact](#-contact)

---

## 📌 Project Overview

This project analyzes Walmart’s sales data and forecasts future weekly sales using machine learning. It includes:

- A Jupyter Notebook that trains multiple models (Random Forest, XGBoost, LightGBM, ...) using a custom time-series pipeline.
- A Streamlit dashboard to explore sales data and compare predictions interactively, allowing users to filter by store, department, time range and also ranking from best to worst based on total sales, $ Growth and % Growth.

---

## 📊 Dataset

### Sources:
- **Model Training**: Historical sales and market data from  `https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview`, `yfinance`, `pandas-datareader`, and `akshare`. Also some columns for holidays and special events or Tax return used a ramp up and down do represent importance.
- **Dashboard**: Pre-processed CSV file `df_wm_store_sales_predictions.csv`, which contains weekly sales and predictions.

| Data Type         | Source                            | Description                                               |
|-------------------|------------------------------------|----------------------------------------------------------|
| Stock Data        | `yfinance` / `pandas-datareader`   | Weekly Walmart stock prices and economic indicators      |
| Processed Dataset | Local CSV (`df_wm_store_sales_predictions.csv`) | Sales & predictions used in the dashboard   |

---

## 🔄 Pipeline & Workflow

### 🧠 Model Training (Jupyter Notebook)
1. **Data Fetching** – Collect WMT stock and macroeconomic indicators
2. **Feature Engineering** – Create time-aware features and lags
3. **Time-Series CV** – `TimeSeriesSplit` with performance tracking
4. **Modeling** – Train Random Forest, XGBoost, LightGBM, ...
5. **Interpretation** – SHAP values and permutation importance ( not completed because of lack of compute power and time constrains

### 📊 Streamlit Dashboard
- `app.py`: Main controller for layout, interaction, routing
- `data_loader.py`: Loads cached data using `@st.cache_data`
- `filters.py`: Applies store/department/date filters
- `metrics.py`: Calculates KPIs (sales totals, growth, date ranges)
- `ui_components.py`: Charts, grids, headers, KPIs, footers

---

## 📈 Insights

- The Random Forest model accurately captures general sales trends.
- SHAP analysis would highlights the most impactful features on predictions.
- Dynamic visual tools make it easy to identify underperforming stores or departments.

### Next Steps

- 🔧 **Hyperparameter Tuning**: Optimize Random Forest for better accuracy  ( if possible get more data on departments and have daily or hourly sales instead of weekly)
- 📊 **More Visuals**: Add SHAP force plots  

---

## 💻 Usage Examples

### Install requirements:

- For notebook usage just ran all cells. If you want to run on kaggle just delete the """ in the first cell. ( note that there is a function called "%%skip". This function was used to run all cell but those that start with that.
- To check streamlit app just run the streamlit.bat found in the main directory.
---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add new analysis"`)
4. Push (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📬 Contact

## Authors

### Jorge M. M. L. Rodrigues  
- **Email:** [jorgemmlrodrigues@gmail.com](mailto:jorgemmlrodrigues@gmail.com)  
- **GitHub:** [github.com/JorgeMMLRodrigues](https://github.com/JorgeMMLRodrigues)

*Feel free to open issues or discussions!*
