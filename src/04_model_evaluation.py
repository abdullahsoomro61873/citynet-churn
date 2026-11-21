
# ============================================================
# LAB 4: MODEL EVALUATION
# Project: CityNet Customer Churn Prediction
# ============================================================


# A. Import required libraries

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Import reusable function from Lab 2
from data_preprocessing import load_and_split_data


# B. Set project directories

# Get the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directory containing saved models
MODELS_DIR = PROJECT_DIR / "models"

# Directory for saving evaluation diagrams
RESULTS_DIR = PROJECT_DIR / "results"

# Create results folder automatically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# C. Load testing data

print("\n========== LOADING TEST DATA ==========")

# Load the same dataset split used in Lab 3
X_train, X_test, y_train, y_test = load_and_split_data()

print("\nTesting data loaded successfully!")
print("Testing records:", len(X_test))


# D. Load trained models

print("\n========== LOADING TRAINED MODELS ==========")

# Load Logistic Regression pipeline
logistic_model = joblib.load(
    MODELS_DIR / "logistic_model.joblib"
)

print("Logistic Regression model loaded!")

# Load Linear SVM pipeline
svm_model = joblib.load(
    MODELS_DIR / "svm_model.joblib"
)

print("Linear SVM model loaded!")


# E. Generate predictions

print("\n========== GENERATING PREDICTIONS ==========")

# Generate predictions using Logistic Regression
logistic_predictions = logistic_model.predict(X_test)

print("Logistic Regression predictions generated!")

# Generate predictions using Linear SVM
svm_predictions = svm_model.predict(X_test)

print("Linear SVM predictions generated!")


# F. Calculate evaluation metrics

print("\n========== CALCULATING METRICS ==========")

# Calculate Logistic Regression metrics
logistic_metrics = {
    "Accuracy": accuracy_score(y_test, logistic_predictions),
    "Precision": precision_score(
        y_test, logistic_predictions,
        pos_label=1, zero_division=0
    ),
    "Recall": recall_score(
        y_test, logistic_predictions,
        pos_label=1, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, logistic_predictions,
        pos_label=1, zero_division=0
    )
}

# Calculate Linear SVM metrics
svm_metrics = {
    "Accuracy": accuracy_score(y_test, svm_predictions),
    "Precision": precision_score(
        y_test, svm_predictions,
        pos_label=1, zero_division=0
    ),
    "Recall": recall_score(
        y_test, svm_predictions,
        pos_label=1, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, svm_predictions,
        pos_label=1, zero_division=0
    )
}

# Display Logistic Regression results
print("\nLOGISTIC REGRESSION")

for metric, value in logistic_metrics.items():
    print(f"{metric}: {value:.4f}")

# Display Linear SVM results
print("\nLINEAR SVM")

for metric, value in svm_metrics.items():
    print(f"{metric}: {value:.4f}")


# G. Generate Confusion Matrix diagrams

print("\n========== CONFUSION MATRICES ==========")

# Calculate Logistic Regression confusion matrix
logistic_cm = confusion_matrix(
    y_test, logistic_predictions,
    labels=[0, 1]
)

# Display Logistic Regression confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=["No Churn", "Churn"]
).plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save diagram
plt.savefig(
    RESULTS_DIR / "logistic_confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()

print("Logistic Regression confusion matrix saved!")


# Calculate Linear SVM confusion matrix
svm_cm = confusion_matrix(
    y_test, svm_predictions,
    labels=[0, 1]
)

# Display Linear SVM confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=["No Churn", "Churn"]
).plot(cmap="Greens", values_format="d")

plt.title("Linear SVM - Confusion Matrix")
plt.tight_layout()

# Save diagram
plt.savefig(
    RESULTS_DIR / "svm_confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()

print("Linear SVM confusion matrix saved!")


# H. Model Performance Comparison Chart

print("\n========== MODEL PERFORMANCE COMPARISON ==========")

# Create a DataFrame containing both models' metrics
comparison_df = pd.DataFrame({
    "Logistic Regression": logistic_metrics,
    "Linear SVM": svm_metrics
})

# Display comparison table
print("\n", comparison_df.round(4))

# Plot comparison bar chart
comparison_df.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.7
)

plt.title("Model Performance Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Models")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save comparison chart
plt.savefig(
    RESULTS_DIR / "model_performance_comparison.png",
    dpi=300
)

plt.show()
plt.close()

print("Model performance comparison saved!")


# I. Classification Report Heatmap

print("\n========== CLASSIFICATION REPORT HEATMAP ==========")

# Generate classification reports for both models
logistic_report = classification_report(
    y_test,
    logistic_predictions,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    output_dict=True,
    zero_division=0
)

svm_report = classification_report(
    y_test,
    svm_predictions,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    output_dict=True,
    zero_division=0
)

# Select precision, recall and F1-score for both classes
report_data = {
    "Logistic - No Churn": [
        logistic_report["No Churn"]["precision"],
        logistic_report["No Churn"]["recall"],
        logistic_report["No Churn"]["f1-score"]
    ],

    "Logistic - Churn": [
        logistic_report["Churn"]["precision"],
        logistic_report["Churn"]["recall"],
        logistic_report["Churn"]["f1-score"]
    ],

    "SVM - No Churn": [
        svm_report["No Churn"]["precision"],
        svm_report["No Churn"]["recall"],
        svm_report["No Churn"]["f1-score"]
    ],

    "SVM - Churn": [
        svm_report["Churn"]["precision"],
        svm_report["Churn"]["recall"],
        svm_report["Churn"]["f1-score"]
    ]
}

# Convert report data into a DataFrame
report_df = pd.DataFrame(
    report_data,
    index=["Precision", "Recall", "F1-Score"]
)

# Create heatmap
plt.figure(figsize=(11, 5))

sns.heatmap(
    report_df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    linewidths=0.5
)

plt.title("Classification Report Heatmap")
plt.tight_layout()

# Save heatmap
plt.savefig(
    RESULTS_DIR / "classification_report_heatmap.png",
    dpi=300
)

plt.show()
plt.close()

print("Classification report heatmap saved!")


# J. ROC Curve Comparison

print("\n========== ROC CURVE COMPARISON ==========")

# Logistic Regression provides probability scores
logistic_scores = logistic_model.predict_proba(X_test)[:, 1]

# Linear SVM provides decision scores
svm_scores = svm_model.decision_function(X_test)

# Calculate Logistic Regression ROC values
logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_scores,
    pos_label=1
)

# Calculate Linear SVM ROC values
svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_scores,
    pos_label=1
)

# Calculate Area Under the Curve (AUC)
logistic_auc = auc(logistic_fpr, logistic_tpr)
svm_auc = auc(svm_fpr, svm_tpr)

# Plot both ROC curves
plt.figure(figsize=(9, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)

plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"Linear SVM (AUC = {svm_auc:.3f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save ROC curve
plt.savefig(
    RESULTS_DIR / "roc_curve_comparison.png",
    dpi=300
)

plt.show()
plt.close()

print("ROC curve comparison saved!")

print(f"Logistic Regression AUC: {logistic_auc:.4f}")
print(f"Linear SVM AUC: {svm_auc:.4f}")


# K. Precision-Recall Curve Comparison

print("\n========== PRECISION-RECALL CURVE ==========")

# Calculate Logistic Regression precision-recall values
logistic_precision_curve, logistic_recall_curve, _ = (
    precision_recall_curve(
        y_test,
        logistic_scores,
        pos_label=1
    )
)

# Calculate Linear SVM precision-recall values
svm_precision_curve, svm_recall_curve, _ = (
    precision_recall_curve(
        y_test,
        svm_scores,
        pos_label=1
    )
)

# Calculate Average Precision scores
logistic_ap = average_precision_score(
    y_test,
    logistic_scores,
    pos_label=1
)

svm_ap = average_precision_score(
    y_test,
    svm_scores,
    pos_label=1
)

# Plot both Precision-Recall curves
plt.figure(figsize=(9, 6))

plt.plot(
    logistic_recall_curve,
    logistic_precision_curve,
    label=f"Logistic Regression (AP = {logistic_ap:.3f})"
)

plt.plot(
    svm_recall_curve,
    svm_precision_curve,
    label=f"Linear SVM (AP = {svm_ap:.3f})"
)

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save Precision-Recall curve
plt.savefig(
    RESULTS_DIR / "precision_recall_curve_comparison.png",
    dpi=300
)

plt.show()
plt.close()

print("Precision-Recall curve saved!")

print(f"Logistic Regression AP: {logistic_ap:.4f}")
print(f"Linear SVM AP: {svm_ap:.4f}")


# L. Final Evaluation Summary

print("\n========== LAB 4 FINAL SUMMARY ==========")

print("Logistic Regression: Evaluation completed")
print("Linear SVM: Evaluation completed")

print("\nTotal testing records:", len(X_test))

print("\nAll six evaluation diagrams saved in:")
print(RESULTS_DIR)

print("\nGenerated diagrams:")

for image_file in sorted(RESULTS_DIR.glob("*.png")):
    print("-", image_file.name)

print("\nLab 4 completed successfully!")
