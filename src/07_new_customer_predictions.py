
# ============================================================
# LAB 7: NEW CUSTOMER CHURN PREDICTIONS
# Project: CityNet Customer Churn Prediction
# ============================================================


# A. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# B. DEFINE PROJECT DIRECTORIES
# ============================================================

# Get the main project directory.
# This file is located inside the src folder.

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define the data directory.

DATA_DIR = PROJECT_DIR / "data"

# Define the models directory.

MODELS_DIR = PROJECT_DIR / "models"

# Define the results directory.

RESULTS_DIR = PROJECT_DIR / "results"

# Create results directory if it does not exist.

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# C. DEFINE FILE PATHS
# ============================================================

# Path of the new customers dataset.

NEW_CUSTOMERS_FILE = DATA_DIR / "new_customers.csv"

# Path of the saved Linear SVM model from Lab 6.

MODEL_FILE = MODELS_DIR / "selected_linear_svm.joblib"

# Path where the final prediction report will be saved.

OUTPUT_FILE = RESULTS_DIR / "new_customer_predictions.csv"


# ============================================================
# D. VERIFY REQUIRED FILES
# ============================================================

print("\n========== VERIFYING REQUIRED FILES ==========")

# Check whether the new customers dataset exists.

if not NEW_CUSTOMERS_FILE.exists():
    raise FileNotFoundError(
        f"New customers dataset not found: {NEW_CUSTOMERS_FILE}"
    )

# Check whether the saved model exists.

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Saved Linear SVM model not found: {MODEL_FILE}"
    )

print("New customers dataset found!")
print("Saved Linear SVM model found!")


# ============================================================
# E. LOAD NEW CUSTOMERS DATASET
# ============================================================

print("\n========== LOADING NEW CUSTOMERS DATA ==========")

# Read the new customers CSV file.

new_customers_df = pd.read_csv(NEW_CUSTOMERS_FILE)

# Display dataset information.

print("Dataset Shape:", new_customers_df.shape)

print("\nDataset Columns:")
print(new_customers_df.columns.tolist())

print("\nFirst 5 Customers:")
print(new_customers_df.head().to_string(index=False))


# ============================================================
# F. SEPARATE CUSTOMER IDs AND MODEL FEATURES
# ============================================================

print("\n========== PREPARING CUSTOMER FEATURES ==========")

# Customer ID is only an identifier.
# It must not be passed to the machine learning model.

if "customer_id" not in new_customers_df.columns:
    raise ValueError("customer_id column is missing!")

# Extract customer IDs and preserve their original order.

customer_ids = new_customers_df["customer_id"].reset_index(drop=True)

# The model expects exactly the same features
# that were used during training in Lab 6.

expected_features = [
    "region",
    "plan",
    "monthly_bill",
    "tv_bundle",
    "tenure_months",
    "support_calls_90d",
    "outage_hours_90d",
    "payment_method",
    "auto_pay",
    "paid_late_last_bill",
    "age"
]

# Check for missing required features.

missing_features = [
    feature
    for feature in expected_features
    if feature not in new_customers_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

# Select features in the exact training order.

X_new = new_customers_df[expected_features].copy()

# Verify the number of features.

print("Customer IDs:", len(customer_ids))
print("Model Features:", X_new.shape[1])

print("\nFeature Columns:")
print(X_new.columns.tolist())

# Display missing values.

print("\nMissing Values Per Feature:")
print(X_new.isnull().sum())


# ============================================================
# G. LOAD SAVED LINEAR SVM MODEL
# ============================================================

print("\n========== LOADING SAVED LINEAR SVM MODEL ==========")

# Load the complete preprocessing and Linear SVM pipeline.
# No model training will happen in Lab 7.

model = joblib.load(MODEL_FILE)

print("Saved model loaded successfully!")

print("Model Type:", type(model))

print("Pipeline Steps:", model.named_steps.keys())


# ============================================================
# H. VERIFY MODEL FEATURE COMPATIBILITY
# ============================================================

print("\n========== VERIFYING MODEL FEATURES ==========")

# Retrieve the feature names recorded by the saved pipeline.

model_features = list(model.feature_names_in_)

# Compare new customer features with training features.

if list(X_new.columns) != model_features:
    raise ValueError(
        "New customer features do not match training features!"
    )

print("Training and new customer features match!")
print("Total Matching Features:", len(model_features))


# ============================================================
# I. GENERATE CHURN PREDICTIONS
# ============================================================

print("\n========== GENERATING CHURN PREDICTIONS ==========")

# Generate churn predictions using the saved model.
#
# 0 = Not predicted to churn
# 1 = Predicted to churn
#
# The saved pipeline automatically applies
# the preprocessing steps before prediction.

predictions = model.predict(X_new)

print("Churn predictions generated successfully!")


# ============================================================
# J. GENERATE CUSTOMER RISK SCORES
# ============================================================

print("\n========== GENERATING CUSTOMER RISK SCORES ==========")

# Calculate continuous decision scores using Linear SVM.
#
# Higher positive score = stronger churn prediction.
# Negative score = prediction is on the non-churn side.
#
# These scores are NOT probability percentages.

risk_scores = model.decision_function(X_new)

print("Risk scores generated successfully!")


# ============================================================
# K. CREATE FINAL PREDICTION REPORT
# ============================================================

print("\n========== CREATING PREDICTION REPORT ==========")

# Combine customer IDs, predictions and risk scores
# into one DataFrame.

prediction_df = pd.DataFrame({

    "customer_id": customer_ids,

    "churn_prediction": predictions,

    "risk_score": risk_scores

})

# Sort customers from highest to lowest risk score.

prediction_df = prediction_df.sort_values(
    by="risk_score",
    ascending=False
).reset_index(drop=True)

# Display the final prediction report.

print("\nNew Customer Churn Predictions:")

print(
    prediction_df.to_string(index=False)
)


# ============================================================
# L. SAVE PREDICTION REPORT
# ============================================================

print("\n========== SAVING PREDICTION REPORT ==========")

# Save the prediction report as a CSV file.

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Prediction report saved successfully!")

print("Output File:", OUTPUT_FILE)


# ============================================================
# M. FINAL VERIFICATION
# ============================================================

print("\n========== FINAL VERIFICATION ==========")

# Reload the saved CSV report.

saved_predictions = pd.read_csv(OUTPUT_FILE)

# Check total number of predicted customers.

total_customers = len(saved_predictions)

# Check whether any prediction is missing.

missing_predictions = (
    saved_predictions["churn_prediction"].isnull().sum()
)

# Check whether any risk score is missing.

missing_risk_scores = (
    saved_predictions["risk_score"].isnull().sum()
)

# Count customers predicted to churn.

predicted_churners = (
    saved_predictions["churn_prediction"] == 1
).sum()

# Count customers not predicted to churn.

predicted_non_churners = (
    saved_predictions["churn_prediction"] == 0
).sum()

# Display final verification results.

print("Total Customers Predicted:", total_customers)

print("Missing Predictions:", missing_predictions)

print("Missing Risk Scores:", missing_risk_scores)

print("Predicted Churners:", predicted_churners)

print("Predicted Non-Churners:", predicted_non_churners)

print("CSV File Exists:", OUTPUT_FILE.exists())

# Verify all completion conditions.

if (
    total_customers == 8
    and missing_predictions == 0
    and missing_risk_scores == 0
    and OUTPUT_FILE.exists()
):
    print("\nLAB 7 COMPLETED SUCCESSFULLY!")

else:
    print("\nLAB 7 VERIFICATION FAILED!")

    raise ValueError(
        "Please check the prediction report and dataset."
    )