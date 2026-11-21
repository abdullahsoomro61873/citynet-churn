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

## 5. Lab 2 – Data Preprocessing

In this lab, we prepared the dataset for machine learning.

### Tasks Completed

- Separated features and target variable.
- Removed customer_id and cancel_request_logged.
- Identified numerical and categorical features.
- Split data into training (80%) and testing (20%).
- Handled missing values using median and mode.
- Converted categorical features using One-Hot Encoding.
- Scaled numerical features using StandardScaler.
- Used pipelines to organize preprocessing steps.

### Data Splitting Results

| Dataset | Records |
|---|---:|
| Training | 4,000 (80%) |
| Testing | 1,000 (20%) |

### Preprocessing Results

| Description | Result |
|---|---:|
| Original Features | 11 |
| Processed Features | 22 |
| Training Churners | 717 |
| Testing Churners | 179 |

The preprocessing pipeline was fitted on training data and applied to testing data to avoid data leakage.

## 6. Lab 3 – Model Training and Saving

### Objective
Train machine learning models to predict customer churn and save trained models for future evaluation and predictions.

### Models Used

1. **Logistic Regression**
   - Used as the baseline model.
   - Trained with numerical feature scaling.

2. **Linear Support Vector Machine (SVM)**
   - Used as the main candidate model.
   - Trained with numerical feature scaling.

### Training Setup

- Dataset: 5,000 customer records
- Training data: 80% (4,000 records)
- Testing data: 20% (1,000 records)
- Random state: 42
- Stratified train-test split
- Missing values handled using imputation.
- Categorical features encoded using One-Hot Encoding.
- Numerical features scaled using StandardScaler.
- Scikit-learn Pipelines used for preprocessing and model training.

### Training Results

- Logistic Regression: Successfully trained.
- Linear SVM: Successfully trained.

### Model Saving

Both trained models were saved using the Joblib library.

Saved model files:

```text
models/
├── logistic_model.joblib
└── svm_model.joblib
```

- Complete preprocessing and model pipelines were saved.
- Models can be loaded later without retraining.
- Saved models will be used for future evaluation and predictions.

### Important Note

- Both models use the same training and testing split.
- Each model has its own preprocessing pipeline.
- Model performance evaluation will be performed in Lab 4.
- Final model selection will be based on evaluation and comparison results.


## 7. How to Run

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

## 8. Model Results

Model performance will be documented after training and evaluation.

| Metric | Baseline | Our Model |
|---|---|---|
| PR-AUC | Pending | Pending |
| Churners in Top 100 | Pending | Pending |

## 9. Intended Use

The model is intended to help the customer retention team prioritize customers who may be at risk of leaving the service.

Predictions should support human decision-making and customer outreach.

## 10. Not for

- Automatically cancelling customer accounts.
- Making final business decisions without human review.
- Treating predictions as guaranteed outcomes.

## 11. Assumptions

- Historical customer information can help identify churn patterns.
- Customer features are available before the prediction date.
- The target variable represents customer churn.
- Predictions are used to prioritize retention efforts.

## 12. Risks and Limitations

- Missing values may affect model performance.
- Historical patterns may change over time.
- The dataset may not represent all real-world customer behaviour.
- Some features may introduce data leakage if they contain information recorded after a customer decides to leave.
- Model predictions are estimates, not guarantees.
