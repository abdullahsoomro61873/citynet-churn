# CityNet Customer Churn Prediction

## 1. Project Overview

CityNet Customer Churn Prediction is a machine learning project that aims to identify customers who are likely to leave a telecom service.

The goal is to help the customer retention team identify high-risk customers and prioritize them for follow-up calls.

## 2. User Story

As a retention manager, I want a weekly list of the 100 customers most likely to cancel their service next month so that my team can contact them first.

## 3. Dataset Overview

The project uses a customer churn dataset containing:

- **Total records:** 5,000 customers
- **Total columns:** 14
- **Target variable:** `churned`
- **Prediction target:** Whether a customer leaves the service.

Target variable values:

- `0` – Customer did not leave.
- `1` – Customer left the service.

## 4. Lab 1 – Dataset Understanding and EDA

In this lab, we explored the dataset to understand its structure, quality, and customer churn patterns.

### Tasks Completed

- Loaded and inspected the dataset.
- Checked dataset dimensions and column names.
- Reviewed data types and missing values.
- Generated statistical summaries.
- Analyzed the target variable.
- Explored categorical and numerical features.
- Performed final data quality checks.

### Data Quality Findings

| Column | Missing Values |
|---|---:|
| `age` | 411 (8.22%) |
| `payment_method` | 149 (2.98%) |
| Other columns | 0 |

Additional checks:

- Total unique customer IDs: 5,000
- Duplicate customer IDs: 0

Missing values will be handled during the preprocessing stage.

### Key Findings

1. The overall customer churn rate is **17.92%**.
2. The Central region has the highest churn rate at **21.21%**.
3. The Fiber 100 plan has the highest churn rate at **21.23%**.
4. Customers using cash payment have the highest churn rate at **27.79%**.
5. Churned customers have a lower average tenure (26.43 months) than non-churned customers (39.43 months).
6. Churned customers also have higher average support calls and outage hours.
7. The `cancel_request_logged` feature has a strong association with churn and needs to be reviewed for potential data leakage.

**Note:** These findings show associations in the dataset. They do not prove that a particular factor directly causes customer churn.

## 5. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train and save the machine learning pipeline:

```bash
python src/lab7_save.py
```

Generate predictions for new customers:

```bash
python src/predict.py data/new_customers.csv
```

The prediction script writes results to:

`reports/predictions.csv`

## 6. Model Results

Model performance will be documented after training and evaluation.

| Metric | Baseline | Our Model |
|---|---|---|
| PR-AUC | Pending | Pending |
| Churners in Top 100 | Pending | Pending |

## 7. Intended Use

The model is intended to help the customer retention team prioritize customers who may be at risk of leaving the service.

Predictions should support human decision-making and customer outreach.

## 8. Not for

- Automatically cancelling customer accounts.
- Making final business decisions without human review.
- Treating predictions as guaranteed outcomes.

## 9. Assumptions

- Historical customer information can help identify churn patterns.
- Customer features are available before the prediction date.
- The target variable represents customer churn.
- Predictions are used to prioritize retention efforts.

## 10. Risks and Limitations

- Missing values may affect model performance.
- Historical patterns may change over time.
- The dataset may not represent all real-world customer behaviour.
- Some features may introduce data leakage if they contain information recorded after a customer decides to leave.
- Model predictions are estimates, not guarantees.
