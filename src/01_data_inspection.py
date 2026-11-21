print("A: Dataset Understanding")
# ==========================================
# CITYNET CHURN PROJECT
# A: Dataset Understanding
# ==========================================

# Step 1: Import Pandas
import pandas as pd

# Step 2: Load the dataset
df = pd.read_csv("data/citynet_churn.csv")

# Step 3: Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Step 4: Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# Step 5: Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 6: Display data types
print("\n========== DATASET INFORMATION ==========")
df.info()

# Step 7: Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Step 8: Display statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all"))

print("B: Analyze Dataset Results")
# ==========================================
# B. Analyze Dataset Results
# ==========================================

# ==========================================
# STEP 9: CHURN DISTRIBUTION
# ==========================================

print("\n========== CHURN DISTRIBUTION ==========")

print(df["churned"].value_counts())

print("\n========== CHURN PERCENTAGE ==========")

print(df["churned"].value_counts(normalize=True) * 100)


# ==========================================
# STEP 10: DUPLICATE RECORDS
# ==========================================

print("\n========== DUPLICATE RECORDS ==========")

print(df.duplicated().sum())


# ==========================================
# STEP 11: UNIQUE VALUES
# ==========================================

print("\n========== UNIQUE VALUES ==========")

categorical_columns = [
    "region",
    "plan",
    "tv_bundle",
    "payment_method",
    "auto_pay",
    "paid_late_last_bill"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))


print("C: Business Analysis")
# ==========================================
# C. Business Analysis
# ==========================================

# ==========================================
# STEP 12: REGION-WISE CHURN ANALYSIS
# ==========================================

print("\n========== REGION-WISE CHURN RATE ==========")

region_churn = df.groupby("region")["churned"].mean() * 100

print(region_churn.round(2))


# ==========================================
# STEP 13: PLAN-WISE CHURN ANALYSIS
# ==========================================

print("\n========== PLAN-WISE CHURN RATE ==========")

plan_churn = df.groupby("plan")["churned"].mean() * 100

print(plan_churn.round(2))


# ==========================================
# STEP 14: PAYMENT-WISE CHURN ANALYSIS
# ==========================================

print("\n========== PAYMENT-WISE CHURN RATE ==========")

payment_churn = df.groupby("payment_method", dropna=False)["churned"].mean() * 100

print(payment_churn.round(2))

print("D: Numerical Feature Analysis")
# ==========================================
# D. Numerical Feature Analysis
# ==========================================

# ==========================================
# STEP 15: NUMERICAL FEATURE ANALYSIS
# ==========================================

print("\n========== NUMERICAL FEATURES BY CHURN ==========")

numerical_columns = [
    "monthly_bill",
    "tenure_months",
    "support_calls_90d",
    "outage_hours_90d",
    "age"
]

print(
    df.groupby("churned")[numerical_columns]
    .mean()
    .round(2)
)


# ==========================================
# STEP 16: CANCELLATION REQUEST ANALYSIS
# ==========================================

print("\n========== CANCELLATION REQUEST ANALYSIS ==========")

print(
    pd.crosstab(
        df["cancel_request_logged"],
        df["churned"],
        normalize="index"
    ).round(3) * 100
)

print("E: Final Data Quality Check")
# ==========================================
# D. Final Data Quality Check
# ==========================================


# ==========================================
# STEP 17: FINAL DATA QUALITY CHECK
# ==========================================

print("\n========== FINAL DATA QUALITY REPORT ==========")

# 1. Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# 2. Missing values percentage
print("\nMissing Values Percentage:")
print((df.isnull().mean() * 100).round(2))

# 3. Numerical column ranges
print("\nNumerical Column Ranges:")

numerical_columns = [
    "monthly_bill",
    "tenure_months",
    "support_calls_90d",
    "outage_hours_90d",
    "age"
]

print(df[numerical_columns].agg(["min", "max"]))

# 4. Target variable validation
print("\nTarget Variable Values:")
print(df["churned"].value_counts(dropna=False))

# 5. Unique customer IDs
print("\nUnique Customer IDs:")
print(df["customer_id"].nunique())

# 6. Duplicate customer IDs
print("\nDuplicate Customer IDs:")
print(df["customer_id"].duplicated().sum())

print("\n========== DATA QUALITY CHECK COMPLETED ==========")
