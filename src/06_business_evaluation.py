
# ============================================================
# LAB 6: BUSINESS EVALUATION
# Project: CityNet Customer Churn Prediction
# ============================================================


# A. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Import reusable preprocessing functions from Lab 2

from data_preprocessing import (
    load_and_split_data,
    create_preprocessor
)


# ============================================================
# B. DEFINE PROJECT DIRECTORIES
# ============================================================

# Get the main project directory

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define results directory

RESULTS_DIR = PROJECT_DIR / "results"

# Create results directory if it does not exist

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# C. LOAD TRAINING AND TESTING DATA
# ============================================================

print("\n========== LOADING DATA ==========")

# Load the same train/test split used in previous labs

X_train, X_test, y_train, y_test = load_and_split_data()

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)


# ============================================================
# D. CREATE SELECTED LINEAR SVM MODEL
# ============================================================

print("\n========== CREATING SELECTED MODEL ==========")

# Create a complete machine learning pipeline
# Preprocessing and model training are handled together

selected_model = Pipeline(steps=[

    ("preprocessor", create_preprocessor()),

    ("model", LinearSVC(
        C=1,
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    ))

])

print("Algorithm: Linear SVM")
print("C: 1")
print("Class Weight: balanced")
print("Max Iterations: 10000")


# ============================================================
# E. TRAIN SELECTED MODEL
# ============================================================

print("\n========== TRAINING SELECTED MODEL ==========")

# Fit the complete pipeline using training data only

selected_model.fit(X_train, y_train)

print("\nSelected Linear SVM trained successfully!")


# ============================================================
# F. GENERATE CUSTOMER RISK SCORES
# ============================================================

print("\n========== GENERATING CUSTOMER RISK SCORES ==========")

# Load original dataset to preserve customer IDs

df = pd.read_csv(
    PROJECT_DIR / "data" / "citynet_churn.csv"
)

# Retrieve customer IDs corresponding to test data indexes

test_customer_ids = df.loc[
    X_test.index,
    "customer_id"
].reset_index(drop=True)

# Generate continuous decision scores using Linear SVM
# Higher positive scores indicate stronger churn predictions
# These scores are NOT probability percentages

risk_scores = selected_model.decision_function(X_test)

# Create a DataFrame containing customer information

risk_df = pd.DataFrame({

    "customer_id": test_customer_ids,

    "actual_churn": y_test.reset_index(drop=True),

    "risk_score": risk_scores

})

# Rank customers from highest to lowest risk score

risk_df = risk_df.sort_values(
    by="risk_score",
    ascending=False
).reset_index(drop=True)

print("\nTop 10 High-Risk Customers:")

print(
    risk_df.head(10).to_string(index=False)
)

print("\nTotal Ranked Customers:", len(risk_df))


# ============================================================
# G. IDENTIFY TOP 100 HIGH-RISK CUSTOMERS
# ============================================================

print("\n========== TOP 100 HIGH-RISK CUSTOMERS ==========")

# Select the 100 customers with the highest risk scores

top_100_customers = risk_df.head(100).copy()

# Add ranking numbers from 1 to 100

top_100_customers.insert(
    0,
    "risk_rank",
    range(1, len(top_100_customers) + 1)
)

# Save the Top 100 report as a CSV file

report_file = (
    RESULTS_DIR / "top_100_high_risk_customers.csv"
)

top_100_customers.to_csv(
    report_file,
    index=False
)

print("\nTop 100 customers identified successfully!")

print("Total High-Risk Customers:", len(top_100_customers))

print("CSV Report Saved At:", report_file)


# ============================================================
# H. BUSINESS EVALUATION OF TOP 100 CUSTOMERS
# ============================================================

print("\n========== BUSINESS EVALUATION ==========")

# Count customers who actually churned

actual_churned = (
    top_100_customers["actual_churn"] == 1
).sum()

# Count customers who did not churn

active_customers = (
    top_100_customers["actual_churn"] == 0
).sum()

# Calculate percentages

churn_percentage = (
    actual_churned / len(top_100_customers)
) * 100

active_percentage = (
    active_customers / len(top_100_customers)
) * 100

# Calculate Precision@100
# It measures the proportion of actual churners in Top 100

precision_at_100 = (
    actual_churned / len(top_100_customers)
)

# Display business evaluation results

print("\nTop 100 Customer Analysis:")

print("Total High-Risk Customers:", len(top_100_customers))

print("Actually Churned Customers:", actual_churned)

print("Active Customers:", active_customers)

print(f"Churn Percentage: {churn_percentage:.2f}%")

print(f"Active Percentage: {active_percentage:.2f}%")

print(f"Precision@100: {precision_at_100:.4f}")


# ============================================================
# I. DISPLAY FINAL BUSINESS SUMMARY
# ============================================================

print("\n========== FINAL BUSINESS SUMMARY ==========")

print("Model: Tuned Linear SVM")

print("Customers Evaluated:", len(risk_df))

print("High-Risk Customers Selected:", len(top_100_customers))

print("Actual Churners in Top 100:", actual_churned)

print("Potential Retention Targets:", active_customers)

print(f"Precision@100: {precision_at_100:.4f}")

print("\nGenerated Report:")

print(report_file)


# ============================================================
# I. DESIGN CUSTOMER RETENTION STRATEGY
# ============================================================

print("\n========== RETENTION STRATEGY ==========")

# Add useful business information to the Top 100 report

business_customers = top_100_customers.merge(

    df[[
        "customer_id",
        "monthly_bill",
        "support_calls_90d",
        "outage_hours_90d",
        "tenure_months",
        "paid_late_last_bill"
    ]],

    on="customer_id",

    how="left"

)

# Only customers who did not churn are potential
# retention campaign targets in this historical analysis.

retention_targets = business_customers[
    business_customers["actual_churn"] == 0
].copy()

# Calculate average monthly bill in the dataset

average_monthly_bill = df["monthly_bill"].mean()

# Use the dataset median as an initial high-bill threshold

high_bill_threshold = df["monthly_bill"].median()


# Define a simple rule-based retention strategy

def recommend_retention_action(customer):

    # Customers experiencing frequent service outages

    if customer["outage_hours_90d"] >= 10:

        return "Service Recovery"

    # Customers contacting support frequently

    elif customer["support_calls_90d"] >= 3:

        return "Priority Customer Support"

    # Customers with a late payment

    elif customer["paid_late_last_bill"] == 1:

        return "Flexible Payment Reminder"

    # Customers with higher monthly bills

    elif customer["monthly_bill"] >= high_bill_threshold:

        return "Special Discount Offer"

    # General loyalty offer

    else:

        return "Loyalty Offer"


# Assign a recommended action to each potential target

retention_targets["retention_action"] = (
    retention_targets.apply(
        recommend_retention_action,
        axis=1
    )
)

# Add retention action to the complete Top 100 report

business_customers = business_customers.merge(

    retention_targets[
        ["customer_id", "retention_action"]
    ],

    on="customer_id",

    how="left"

)

# Save the enhanced business report

business_customers.to_csv(
    RESULTS_DIR / "top_100_high_risk_customers.csv",
    index=False
)

# Display retention strategy results

print("Potential Retention Targets:", len(retention_targets))

print("\nRecommended Retention Actions:")

print(
    retention_targets["retention_action"]
    .value_counts()
    .to_string()
)

print("\nUpdated Top 100 Business Report Saved!")

# ============================================================
# J. ESTIMATE POTENTIAL BUSINESS IMPACT
# ============================================================

print("\n========== BUSINESS IMPACT ANALYSIS ==========")

# Business scenario assumptions.
# These are illustrative assumptions, not measured results.

DISCOUNT_RATE = 0.10

DISCOUNT_MONTHS = 3

EXPECTED_RETENTION_RATE = 0.30

RETAINED_REVENUE_MONTHS = 6

GROSS_MARGIN = 0.60


# Number of potential retention targets

target_count = len(retention_targets)

# Calculate average monthly bill of potential targets

if target_count > 0:

    target_average_bill = retention_targets[
        "monthly_bill"
    ].mean()

else:

    target_average_bill = 0


# Estimate the number of customers retained

expected_retained_customers = (
    target_count * EXPECTED_RETENTION_RATE
)


# Estimate revenue from retained customers

expected_saved_revenue = (
    expected_retained_customers
    * target_average_bill
    * RETAINED_REVENUE_MONTHS
)


# Estimate gross contribution from retained customers

expected_gross_contribution = (
    expected_saved_revenue * GROSS_MARGIN
)


# Estimate discount cost.
# Simplified assumption: only successfully retained
# customers receive the discount for three months.

estimated_discount_cost = (
    expected_retained_customers
    * target_average_bill
    * DISCOUNT_RATE
    * DISCOUNT_MONTHS
)


# Calculate estimated net business benefit

estimated_net_benefit = (
    expected_gross_contribution
    - estimated_discount_cost
)


# Display business assumptions

print("\nBUSINESS ASSUMPTIONS:")

print(f"Discount Rate: {DISCOUNT_RATE * 100:.0f}%")

print(f"Discount Duration: {DISCOUNT_MONTHS} months")

print(
    f"Expected Retention Rate: "
    f"{EXPECTED_RETENTION_RATE * 100:.0f}%"
)

print(f"Retained Revenue Period: {RETAINED_REVENUE_MONTHS} months")

print(f"Gross Margin: {GROSS_MARGIN * 100:.0f}%")


# Display estimated financial impact

print("\nBUSINESS IMPACT RESULTS:")

print("Potential Retention Targets:", target_count)

print(
    f"Average Monthly Bill: "
    f"{target_average_bill:.2f}"
)

print(
    f"Expected Retained Customers: "
    f"{expected_retained_customers:.2f}"
)

print(
    f"Expected Saved Revenue: "
    f"{expected_saved_revenue:.2f}"
)

print(
    f"Expected Gross Contribution: "
    f"{expected_gross_contribution:.2f}"
)

print(
    f"Estimated Discount Cost: "
    f"{estimated_discount_cost:.2f}"
)

print(
    f"Estimated Net Business Benefit: "
    f"{estimated_net_benefit:.2f}"
)


