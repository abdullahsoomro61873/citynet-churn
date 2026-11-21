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

# Lab 4 – Model Evaluation

## CityNet Customer Churn Prediction

### 1. Overview

Lab 4 focuses on evaluating two trained Machine Learning models:

- Logistic Regression
- Linear Support Vector Machine (SVM)

The purpose of this lab is to measure model performance using different classification metrics and visualizations.

Both models are evaluated using the same testing dataset containing 1,000 customer records.

### 2. Objectives

- Load previously trained Machine Learning models.
- Generate predictions on testing data.
- Calculate classification evaluation metrics.
- Generate and analyze confusion matrices.
- Compare model performance.
- Visualize classification reports using heatmaps.
- Evaluate models using ROC and Precision-Recall curves.
- Save evaluation diagrams for future reference.

### 3. Models Evaluated

| Model | Description |
|---|---|
| Logistic Regression | A classification algorithm used to predict customer churn. |
| Linear SVM | A Support Vector Machine classifier used to distinguish churn and non-churn customers. |

### 4. Dataset Information

| Description | Value |
|---|---:|
| Total Records | 5,000 |
| Total Features | 11 |
| Training Records | 4,000 |
| Testing Records | 1,000 |
| Target Variable | Customer Churn |

Both models were evaluated using the same testing dataset.

### 5. Evaluation Metrics

The following metrics were used:

**Accuracy:** Measures the overall percentage of correct predictions.

**Precision:** Measures how many predicted churn customers actually churned.

**Recall:** Measures how many actual churn customers were correctly identified.

**F1-Score:** Provides a balance between precision and recall.

**ROC-AUC:** Measures the model's ability to distinguish between churn and non-churn customers.

**Average Precision (AP):** Summarizes performance across different precision-recall thresholds.

### 6. Model Evaluation Results

The following results were obtained during testing:

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 82.50% | 82.40% |
| Precision | 55.88% | 66.67% |
| Recall | 10.61% | 3.35% |
| F1-Score | 17.84% | 6.38% |
| ROC-AUC | 72.47% | 72.35% |
| Average Precision | 40.18% | 39.98% |

### 7. Evaluation Visualizations

All six evaluation diagrams were generated and saved in the `results/` directory.

#### 7.1 Logistic Regression – Confusion Matrix

![Logistic Regression Confusion Matrix](results/logistic_confusion_matrix.png)

#### 7.2 Linear SVM – Confusion Matrix

![Linear SVM Confusion Matrix](results/svm_confusion_matrix.png)

#### 7.3 Model Performance Comparison

![Model Performance Comparison](results/model_performance_comparison.png)

#### 7.4 Classification Report Heatmap

![Classification Report Heatmap](results/classification_report_heatmap.png)

#### 7.5 ROC Curve Comparison

![ROC Curve Comparison](results/roc_curve_comparison.png)

#### 7.6 Precision-Recall Curve Comparison

![Precision Recall Curve Comparison](results/precision_recall_curve_comparison.png)

### 8. Results Analysis

Based on the evaluation results:

- Logistic Regression achieved 82.50% accuracy.
- Linear SVM achieved 82.40% accuracy.
- Linear SVM achieved higher precision (66.67%).
- Logistic Regression achieved higher recall (10.61%).
- Logistic Regression achieved a higher F1-score (17.84%).
- Both models achieved similar ROC-AUC and Average Precision scores.

**Important Observation:**

Although both models achieved approximately 82% accuracy, their recall scores were very low.

This indicates that both models missed a significant number of actual churn customers.

This may be related to class imbalance in the dataset and requires further investigation.

Therefore, accuracy alone is not sufficient to determine the best churn prediction model.

### 9. Project Structure

```text
citynet-churn/
│
├── data/
│
├── models/
│   ├── logistic_model.joblib
│   └── svm_model.joblib
│
├── src/
│   ├── data_preprocessing.py
│   └── 04_model_evaluation.py
│
├── results/
│   ├── logistic_confusion_matrix.png
│   ├── svm_confusion_matrix.png
│   ├── model_performance_comparison.png
│   ├── classification_report_heatmap.png
│   ├── roc_curve_comparison.png
│   └── precision_recall_curve_comparison.png
│
├── requirements.txt
└── README.md
```
**Lab 5 – Model Experiments and Comparison**

The next lab will focus on:

- Investigating class imbalance.
- Experimenting with model configurations.
- Exploring techniques to improve churn detection.
- Comparing model performance.
- Identifying a suitable model for customer churn prediction.

---

**Project:** CityNet Customer Churn Prediction  
**Lab:** 04 – Model Evaluation  
**Status:** Completed

## Lab 5 – Model Improvement & Comparison

### Objective
The objective of Lab 5 was to improve customer churn prediction by experimenting with different machine learning models and comparing their performance using multiple evaluation metrics.

### Models Experimented

Five model configurations were evaluated:

1. Logistic Regression – Baseline model.
2. Linear SVM with Feature Scaling.
3. Linear SVM without Feature Scaling.
4. Random Forest with class balancing.
5. Tuned Linear SVM using GridSearchCV.

### Hyperparameter Tuning

GridSearchCV was used to find the best Linear SVM parameters.

- Best C: `1`
- Best class_weight: `balanced`
- Cross-validation: `5-fold StratifiedKFold`
- Best Cross-Validation F1-score: `0.4262`

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8250 | 0.5588 | 0.1061 | 0.1784 | **0.4018** |
| Linear SVM (Scaled) | 0.8240 | 0.6667 | 0.0335 | 0.0638 | 0.3998 |
| Linear SVM (Unscaled) | 0.8250 | 0.7500 | 0.0335 | 0.0642 | 0.3953 |
| Random Forest | 0.7830 | 0.3443 | 0.2346 | 0.2791 | 0.3183 |
| Tuned Linear SVM | 0.6660 | 0.2966 | **0.6313** | **0.4036** | 0.4014 |

### Evaluation Metrics

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** How many customers predicted as churners actually churned.
- **Recall:** How many actual churners the model successfully identified.
- **F1-Score:** Harmonic balance between Precision and Recall.
- **PR-AUC (AP):** Measures the model's precision-recall performance across different classification thresholds.

### Final Model Selection

The Tuned Linear SVM was selected as the provisional candidate for business evaluation.

Reasons:

- Highest Recall: `0.6313`
- Highest F1-Score: `0.4036`
- PR-AUC (AP): `0.4014`

Logistic Regression achieved the highest PR-AUC (AP) of `0.4018`. However, the difference between the two models was very small (`0.0004`).

Since the business objective is to identify as many potential churn customers as possible, Tuned Linear SVM is the more suitable candidate for the next business evaluation stage.

### Output

Model comparison results are saved in:

`results/model_comparison.csv`

### Key Learning

This lab demonstrated that accuracy alone is not sufficient for evaluating models on an imbalanced dataset. Recall, F1-Score, and PR-AUC provide additional insight into churn prediction performance.

**Next:** Lab 6 – Business Evaluation: Identifying the Top 100 High-Risk Customers and evaluating a customer retention strategy.


### 10. How to Run

**Step 1: Activate the virtual environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Step 2: Install dependencies**

```powershell
pip install -r requirements.txt
```

**Step 3: Run Lab 4**

```powershell
python src/04_model_evaluation.py
```

The evaluation results will be displayed in the terminal, and all six diagrams will be saved automatically in the `results/` directory.

**Note:** Lab 4 requires the trained model files from Lab 3 and the preprocessing module from Lab 2.

### 11. Conclusion

In this lab, Logistic Regression and Linear SVM were evaluated using multiple classification metrics and visualizations.

Both models demonstrated similar overall accuracy and ROC-AUC performance. However, their low recall scores indicate that churn detection needs further improvement.

The evaluation results provide a foundation for the next stage of the project.

### 12. Next Steps


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
