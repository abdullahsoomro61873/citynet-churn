
# =====================================================
# LAB 3: MODEL TRAINING
# Project: CityNet Customer Churn Prediction
# =====================================================

# A. Import required libraries

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Import reusable preprocessing functions from Lab 2
from data_preprocessing import (
    load_and_split_data,
    create_preprocessor
)


# B. Load training and testing data

print("========== LOADING DATA ==========")

X_train, X_test, y_train, y_test = load_and_split_data()

print("\nDataset loaded successfully!")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# C. Create Logistic Regression pipeline

# Pipeline performs:
# 1. Missing value handling
# 2. Categorical encoding
# 3. Numerical scaling
# 4. Logistic Regression training

logistic_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("model", LogisticRegression(max_iter=1000))
])


# D. Train Logistic Regression

print("\n========== LOGISTIC REGRESSION ==========")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")


# E. Create Linear SVM pipeline

# Pipeline performs the same preprocessing
# followed by Linear SVM model training.

svm_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("model", LinearSVC(
        random_state=42,
        max_iter=10000
    ))
])


# F. Train Linear SVM

print("\n========== LINEAR SVM ==========")

svm_model.fit(X_train, y_train)

print("Linear SVM training completed!")


# G. Save trained models

print("\n========== SAVING MODELS ==========")

# Create models folder in project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

# Create folder automatically if it does not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save complete Logistic Regression pipeline
joblib.dump(
    logistic_model,
    MODELS_DIR / "logistic_model.joblib"
)

print("Logistic Regression model saved successfully!")

# Save complete Linear SVM pipeline
joblib.dump(
    svm_model,
    MODELS_DIR / "svm_model.joblib"
)

print("Linear SVM model saved successfully!")


# H. Final training summary

print("\n========== LAB 3 SUMMARY ==========")

print("Logistic Regression: Trained and saved")
print("Linear SVM: Trained and saved")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nSaved model files:")
print("models/logistic_model.joblib")
print("models/svm_model.joblib")

print("\nLab 3 completed successfully!")
