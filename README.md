# 🛒 Olist E-Commerce — End-to-End Analysis

A comprehensive data science project on the **Brazilian Olist E-Commerce dataset**, covering the full analytics lifecycle: data wrangling, exploratory analysis, customer segmentation, machine learning, and revenue forecasting.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Analyses](#key-analyses)
- [Machine Learning Models](#machine-learning-models)
- [Revenue Forecasting](#revenue-forecasting)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Results Summary](#results-summary)

---

## Project Overview

This end-to-end notebook answers critical business questions for an e-commerce platform:

- **When** do customers shop, and which states drive the most orders?
- **Which product categories** generate the highest revenue?
- **Can we predict** whether an order will arrive late — before it ships?
- **What drives** a customer's review score?
- **Which customers** are at risk of churning?
- **What will revenue look like** over the next 6 months?

---

## Dataset

The project uses the publicly available [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle, comprising **6 CSV files** joined into a single master dataframe:

| File | Description |
|---|---|
| `olist_orders_dataset.csv` | Order metadata and timestamps |
| `olist_order_items_dataset.csv` | Items, prices, and freight values |
| `olist_customers_dataset.csv` | Customer location data |
| `olist_products_dataset.csv` | Product dimensions and categories |
| `olist_order_payments_dataset.csv` | Payment type and installments |
| `olist_order_reviews_dataset.csv` | Customer review scores and comments |

---

## Project Structure

```
Olist_ECommerce_End_to_End_Analysis.ipynb
│
├── Chapter 1 — Data Loading & Merging
├── Chapter 2 — Data Cleaning & Missing Value Handling
├── Chapter 3 — Outlier Detection & Business Anomalies
├── Chapter 4 — Feature Engineering (time-based features)
├── Chapter 5 — Exploratory Data Analysis (EDA)
├── Chapter 6 — RFM Customer Segmentation
├── Chapter 7 — Monthly Revenue Trend Analysis
├── Chapter 8 — ML Model 1: Delivery Delay Prediction
├── Chapter 9 — ML Model 2: Review Score Prediction
├── Chapter 10 — ML Model 3: Customer Churn Prediction
└── Chapter 11 — Revenue Forecasting with Prophet
```

---

## Key Analyses

### 🕐 Peak Shopping Hours
Area chart of order volume by hour, revealing Brazil's busiest shopping windows.

### 📦 Delivery Performance
Histogram with box plot overlay showing how often orders arrive early vs. late relative to the estimated delivery date.

### 🏆 Top Revenue Categories
Horizontal bar chart of the top 10 product categories by total revenue.

### 🗺️ Geographic Distribution
Choropleth map and bar chart of order volume by Brazilian state.

### 👥 RFM Customer Segmentation
Customers scored on **Recency, Frequency, and Monetary** value, then classified into segments:

| Segment | Description |
|---|---|
| **Champion** | Recent, frequent, high-spending |
| **Loyal Customer** | High R & F scores |
| **Recent Customer** | Just purchased, not yet frequent |
| **At Risk** | Declining engagement |
| **Potential Loyalist** | Moderate activity |
| **Lost** | Long inactive |

---

## Machine Learning Models

Three XGBoost classifiers are trained and benchmarked against Logistic Regression and Random Forest baselines.

### Model 1 — Delivery Delay Prediction
> *"Will this order arrive late?"*

- **Target:** Binary — Late (1) vs. On Time (0)
- **Features:** Product dimensions, freight value, price, purchase seasonality, customer state, product category
- **Best model:** XGBoost
- **Output:** Probability of late delivery for any new order

### Model 2 — Review Score Prediction
> *"What star rating will this customer leave?"*

- **Target:** Multi-class (1–5 stars)
- **Features:** Delivery time, early/late delta, price, freight value, product weight, payment installments, seasonality, category
- **Best model:** XGBoost

### Model 3 — Customer Churn Prediction
> *"Which customers are about to leave?"*

- **Target:** Binary — Churned (inactive > 180 days) vs. Active
- **Features:** RFM metrics (Recency, Frequency, Monetary), average review score, average delivery days
- **Best model:** XGBoost with ROC-AUC evaluation
- **Output:** Per-customer churn probability + risk tier (Low / Medium / High)

---

## Revenue Forecasting

A **Facebook Prophet** model is fitted to the monthly revenue time series and used to generate a **6-month forward forecast** with a 95% confidence interval.

- Seasonality mode: multiplicative (accounts for revenue growth over time)
- Yearly seasonality enabled
- Trend + seasonality components visualised separately

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `plotly` | Interactive visualisations |
| `scikit-learn` | Preprocessing, baseline models, evaluation |
| `xgboost` | Gradient boosted classifiers |
| `prophet` | Time-series forecasting |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<Lifewitdata>/olist-ecommerce-analysis.git
cd olist-ecommerce-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy plotly scikit-learn xgboost prophet
```

### 3. Download the dataset
Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files in the project root directory.

### 4. Run the notebook
```bash
jupyter notebook Olist_ECommerce_End_to_End_Analysis.ipynb
```

---

## Results Summary

| Model | Algorithm | Metric |
|---|---|---|
| Delivery Delay | XGBoost | Accuracy (see notebook) |
| Review Score | XGBoost | Multi-class accuracy (see notebook) |
| Churn Prediction | XGBoost | AUC-ROC (see notebook) |
| Revenue Forecast | Prophet | 6-month projection with 95% CI |

> Exact metric values are printed in the notebook output cells after training.

---

## 📄 License

This project is released under the MIT License. The Olist dataset is publicly available on Kaggle under its own terms.
