
# ============================================================
# LAB 5: MODEL EXPERIMENTS & COMPARISON
# Project: CityNet Customer Churn Prediction
# ============================================================


# A. Import required libraries

from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Import reusable preprocessing functions

from data_preprocessing import (
    load_and_split_data,
    create_preprocessor
)


# B. Define project directories

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# C. Load training and testing data

print("\n========== LOADING DATA ==========")

X_train, X_test, y_train, y_test = load_and_split_data()

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())


# D. Create a list to store experiment results

experiment_results = []


# E. Define a reusable evaluation function

def evaluate_model(model, model_name):

    # Generate final class predictions (0 or 1)

    y_pred = model.predict(X_test)

    # Get continuous scores for the positive class (churn = 1).
    # These scores are required for Precision-Recall analysis.

    if hasattr(model, "predict_proba"):

        # Logistic Regression and Random Forest
        # Return the probability of churn (class 1).

        positive_class_index = list(model.classes_).index(1)

        y_scores = model.predict_proba(X_test)[
            :, positive_class_index
        ]

    elif hasattr(model, "decision_function"):

        # Linear SVM models
        # Return decision scores instead of 0/1 predictions.

        y_scores = model.decision_function(X_test)

    else:

        raise ValueError(
            f"{model_name} does not support probability "
            "or decision scores."
        )

    # Calculate standard evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate Average Precision (AP).
    # We will report this as PR-AUC (AP).
    # It evaluates precision-recall performance across thresholds.

    pr_auc = average_precision_score(
        y_test,
        y_scores
    )

    # Store all results in a dictionary

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC (AP)": pr_auc
    }

    experiment_results.append(results)

    # Display individual model results

    print("\nModel:", model_name)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"PR-AUC (AP): {pr_auc:.4f}")


# ============================================================
# EXPERIMENT 1: LOGISTIC REGRESSION BASELINE
# ============================================================

print("\n\n========== EXPERIMENT 1 ==========")
print("LOGISTIC REGRESSION BASELINE")

# Create Logistic Regression pipeline

logistic_model = Pipeline(steps=[

    ("preprocessor", create_preprocessor()),

    ("model", LogisticRegression(
        max_iter=1000
    ))

])

# Train model

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")

# Evaluate model

evaluate_model(
    logistic_model,
    "Logistic Regression"
)


# ============================================================
# EXPERIMENT 2: LINEAR SVM WITH SCALING
# ============================================================

print("\n\n========== EXPERIMENT 2 ==========")
print("LINEAR SVM WITH SCALING")

# Create Linear SVM pipeline

svm_scaled = Pipeline(steps=[

    ("preprocessor", create_preprocessor()),

    ("model", LinearSVC(
        random_state=42,
        max_iter=10000
    ))

])

# Train model

svm_scaled.fit(X_train, y_train)

print("Scaled Linear SVM training completed!")

# Evaluate model

evaluate_model(
    svm_scaled,
    "Linear SVM (With Scaling)"
)


# ============================================================
# EXPERIMENT 3: LINEAR SVM WITHOUT SCALING
# ============================================================

print("\n\n========== EXPERIMENT 3 ==========")
print("LINEAR SVM WITHOUT SCALING")

# Reuse existing preprocessing pipeline.
# Replace StandardScaler with passthrough.
# Missing value handling and encoding remain unchanged.

preprocessor_without_scaling = create_preprocessor()

preprocessor_without_scaling.set_params(
    num__scaler="passthrough"
)

# Create SVM pipeline

svm_no_scaling = Pipeline(steps=[

    ("preprocessor", preprocessor_without_scaling),

    ("model", LinearSVC(
        random_state=42,
        max_iter=10000
    ))

])

# Train model

svm_no_scaling.fit(X_train, y_train)

print("Unscaled Linear SVM training completed!")

# Evaluate model

evaluate_model(
    svm_no_scaling,
    "Linear SVM (Without Scaling)"
)


# ============================================================
# EXPERIMENT 4: RANDOM FOREST
# ============================================================

print("\n\n========== EXPERIMENT 4 ==========")
print("RANDOM FOREST")

# Random Forest does not require numerical scaling.
# Therefore, StandardScaler is disabled.

rf_preprocessor = create_preprocessor()

rf_preprocessor.set_params(
    num__scaler="passthrough"
)

# Create Random Forest pipeline

random_forest_model = Pipeline(steps=[

    ("preprocessor", rf_preprocessor),

    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))

])

# Train model

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed!")

# Evaluate model

evaluate_model(
    random_forest_model,
    "Random Forest"
)


# ============================================================
# EXPERIMENT 5: SVM HYPERPARAMETER TUNING
# ============================================================

print("\n\n========== EXPERIMENT 5 ==========")
print("SVM HYPERPARAMETER TUNING")

# Create a fresh SVM pipeline with scaling

svm_tuning_pipeline = Pipeline(steps=[

    ("preprocessor", create_preprocessor()),

    ("model", LinearSVC(
        random_state=42,
        max_iter=10000
    ))

])

# Define hyperparameters to investigate

param_grid = {

    "model__C": [0.01, 0.1, 1, 10],

    "model__class_weight": [None, "balanced"]

}

# Stratified cross-validation.
# Each fold maintains a similar class distribution.

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Create GridSearchCV

grid_search = GridSearchCV(

    estimator=svm_tuning_pipeline,

    param_grid=param_grid,

    scoring="f1",

    cv=cv_strategy,

    n_jobs=-1,

    verbose=1

)

# Train and investigate different configurations.
# Only training data is used during cross-validation.

grid_search.fit(X_train, y_train)

# Get the best model

best_svm_model = grid_search.best_estimator_

print("\nBest SVM Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(f"{grid_search.best_score_:.4f}")

print("\nSVM hyperparameter tuning completed!")

# Evaluate tuned SVM on held-out test data

evaluate_model(
    best_svm_model,
    "Tuned Linear SVM"
)


# ============================================================
# F. CREATE FINAL MODEL COMPARISON TABLE
# ============================================================

print("\n\n========== FINAL MODEL COMPARISON ==========")

# Convert experiment results into DataFrame

comparison_df = pd.DataFrame(experiment_results)

# Keep the original ranking by F1-Score

comparison_df = comparison_df.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

# Display comparison table

print("\n", comparison_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format,
        "PR-AUC (AP)": "{:.4f}".format
    }
))


# ============================================================
# G. SAVE EXPERIMENT RESULTS
# ============================================================

print("\n\n========== SAVING EXPERIMENT RESULTS ==========")

# Save updated comparison table as CSV

results_file = RESULTS_DIR / "model_comparison.csv"

comparison_df.to_csv(
    results_file,
    index=False
)

print("Model comparison saved successfully!")

print("File location:", results_file)


# ============================================================
# H. IDENTIFY TOP EXPERIMENTAL CANDIDATES
# ============================================================

print("\n\n========== EXPERIMENTAL SUMMARY ==========")

# Find the highest F1-Score model

top_f1_experiment = comparison_df.loc[
    comparison_df["F1-Score"].idxmax()
]

# Find the highest PR-AUC model

top_pr_auc_experiment = comparison_df.loc[
    comparison_df["PR-AUC (AP)"].idxmax()
]

print("\nTop candidate based on F1-Score:")

print("Model:", top_f1_experiment["Model"])
print(f"F1-Score: {top_f1_experiment['F1-Score']:.4f}")
print(f"Recall:   {top_f1_experiment['Recall']:.4f}")
print(f"PR-AUC:   {top_f1_experiment['PR-AUC (AP)']:.4f}")


print("\nTop candidate based on PR-AUC (AP):")

print("Model:", top_pr_auc_experiment["Model"])
print(f"PR-AUC:   {top_pr_auc_experiment['PR-AUC (AP)']:.4f}")
print(f"Recall:   {top_pr_auc_experiment['Recall']:.4f}")
print(f"F1-Score: {top_pr_auc_experiment['F1-Score']:.4f}")


print("\nIMPORTANT:")
print("F1-Score and PR-AUC measure different aspects")
print("of model performance.")
print("Do not select a production model using only one metric.")
print("Consider the business objective and retention cost too.")
print("These test-set comparisons are exploratory.")

print("\n========== LAB 5 COMPLETED SUCCESSFULLY ==========")
