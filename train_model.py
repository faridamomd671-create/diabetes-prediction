"""Train the diabetes models, compare them, and save the best one.

Run:  python train_model.py
Output: models/best_model.joblib  +  models/metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# In this dataset, 0 in these columns really means "missing value"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv(DATA_PATH)
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(probability=True, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
    })
    fitted[name] = (pipe, pred)
    print(results[-1])

best = max(results, key=lambda r: r["F1"])["Algorithm"]
best_pipe, best_pred = fitted[best]
print(f"\nBest model (by F1): {best}")

joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump({
        "best_model": best,
        "results": results,
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "features": list(X.columns),
    }, f, indent=2)
print("Saved models/best_model.joblib and models/metrics.json")
