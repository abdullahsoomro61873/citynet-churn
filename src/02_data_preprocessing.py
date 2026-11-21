
# =====================================================
# LAB 2: DATA PREPROCESSING
# Project: CityNet Customer Churn Prediction
# =====================================================

# A. Import required libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# B. Load dataset

df = pd.read_csv("data/citynet_churn.csv")

print("========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)


# C. Separate features and target

# Remove customer_id because it is only an identifier.
# Remove cancel_request_logged for now because it
# may cause data leakage for next-month prediction.

X = df.drop(
    columns=["customer_id", "churned", "cancel_request_logged"]
)

y = df["churned"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# D. Identify numerical and categorical columns

numerical_features = [
    "monthly_bill",
    "tenure_months",
    "support_calls_90d",
    "outage_hours_90d",
    "age"
]

categorical_features = [
    "region",
    "plan",
    "tv_bundle",
    "payment_method",
    "auto_pay",
    "paid_late_last_bill"
]

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# E. Split data into training and testing sets

# 80% data for training
# 20% data for testing
# stratify maintains the churn distribution.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLITTING ==========")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)


# F. Create numerical preprocessing pipeline

# Missing numerical values -> Median
# Scale numerical features -> StandardScaler

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# G. Create categorical preprocessing pipeline

# Missing categorical values -> Most frequent value
# Convert categories into numerical values -> OneHotEncoder

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# H. Combine both preprocessing pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# I. Fit preprocessing on training data only

# Learn missing-value rules, categories and scaling
# from training data.

X_train_processed = preprocessor.fit_transform(X_train)

# Apply the learned transformations to test data.
# Do not fit again on testing data.

X_test_processed = preprocessor.transform(X_test)


# J. Final preprocessing verification

print("\n========== FINAL PREPROCESSING RESULTS ==========")

print("Original Training Shape:", X_train.shape)
print("Processed Training Shape:", X_train_processed.shape)

print("Original Testing Shape:", X_test.shape)
print("Processed Testing Shape:", X_test_processed.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nMissing Values After Preprocessing:")
print("Numerical and categorical missing values handled.")

print("\n========== LAB 2 PREPROCESSING COMPLETED ==========")
