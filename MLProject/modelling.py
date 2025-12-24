"""
Telco Churn Classification with MLflow
Author: Gesang
Dataset: Telco Customer Churn
"""

import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ===============================
# MLflow Configuration
# ===============================
if os.getenv('GITHUB_ACTIONS'):
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running in GitHub Actions - using local file tracking")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running locally - using local file tracking")

mlflow.set_experiment("telco-churn-classification")

# ===============================
# Load Dataset
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "namadataset_preprocessing",
    "TelcoCustomerChurn_preprocessing.csv"
)
print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Training Function
# ===============================
def train_model(n_estimators, random_state):
    """Train RandomForestClassifier with MLflow autolog"""
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log manual metrics (optional)
    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("f1_manual", f1)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train_model(n_estimators=args.n_estimators, random_state=args.random_state)
    print("=== TRAINING COMPLETED SUCCESSFULLY ===")
