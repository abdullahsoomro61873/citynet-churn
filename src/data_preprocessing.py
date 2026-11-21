
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


# B. Define numerical and categorical features

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


# C. Load dataset and split into training/testing data

def load_and_split_data():

    # Load the original dataset
    df = pd.read_csv("data/citynet_churn.csv")

    print("========== DATASET INFORMATION ==========")
    print("Dataset Shape:", df.shape)

    # Separate input features and target variable
    # Remove customer_id because it is only an identifier.
    # Remove cancel_request_logged to avoid data leakage.

    X = df.drop(
        columns=["customer_id", "churned", "cancel_request_logged"]
    )

    y = df["churned"]

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

    # Split data into 80% training and 20% testing.
    # Stratify maintains similar churn distribution.

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

    return X_train, X_test, y_train, y_test


# D. Create reusable preprocessing pipeline

def create_preprocessor():

    # Numerical preprocessing:
    # 1. Fill missing values using median.
    # 2. Scale numerical features using StandardScaler.

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing:
    # 1. Fill missing values using the most frequent value.
    # 2. Convert categories into numerical values.

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine numerical and categorical preprocessing.

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


# E. Run Lab 2 independently when this file is executed directly

if __name__ == "__main__":

    # Load and split the dataset
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply learned preprocessing to testing data
    X_test_processed = preprocessor.transform(X_test)

    # Display final results
    print("\n========== FINAL PREPROCESSING RESULTS ==========")

    print("Original Training Shape:", X_train.shape)
    print("Processed Training Shape:", X_train_processed.shape)

    print("Original Testing Shape:", X_test.shape)
    print("Processed Testing Shape:", X_test_processed.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    print("\n========== LAB 2 PREPROCESSING COMPLETED ==========")
