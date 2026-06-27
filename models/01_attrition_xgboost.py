import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import json

load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

def log(msg): print(f"[XGB] {msg}")

# ── 1. Load data from star schema ─────────────────────────────────────────────

def load_data():
    log("Loading employee data from Postgres...")
    query = """
        SELECT
            fe.age, fe.gender, fe.monthly_income, fe.job_satisfaction,
            fe.years_at_company, fe.overtime, fe.distance_from_home,
            fe.education, fe.environment_satisfaction, fe.performance_rating,
            dd.department_name, dd.job_role,
            dal.attrition
        FROM fact_employees fe
        JOIN dim_department dd ON fe.department_id = dd.department_id
        JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
    """
    df = pd.read_sql(query, engine)
    log(f"  → {len(df)} rows loaded")
    return df

# ── 2. Feature engineering ────────────────────────────────────────────────────

def prepare_features(df):
    log("Preparing features...")

    # encode categoricals
    le = LabelEncoder()
    for col in ["gender", "overtime", "department_name", "job_role"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # target
    df["target"] = (df["attrition"] == "Yes").astype(int)

    feature_cols = [
        "age", "gender", "monthly_income", "job_satisfaction",
        "years_at_company", "overtime", "distance_from_home",
        "education", "environment_satisfaction", "performance_rating",
        "department_name", "job_role"
    ]

    X = df[feature_cols]
    y = df["target"]

    log(f"  → {X.shape[1]} features, {y.sum()} attrition cases out of {len(y)}")
    return X, y, feature_cols

# ── 3. Train XGBoost ──────────────────────────────────────────────────────────

def train_model(X, y):
    log("Training XGBoost classifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale_pos_weight handles class imbalance (more "No" than "Yes")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos
    log(f"  → Class imbalance ratio: {scale:.2f} (used as scale_pos_weight)")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model, X_train, X_test, y_train, y_test

# ── 4. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, X_test, y_test, feature_cols):
    log("Evaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    log(f"  → Accuracy:  {acc:.4f}")
    log(f"  → ROC-AUC:   {roc:.4f}")

    # 5-fold cross validation on full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    log(f"  → CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

    # feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\nTop 5 attrition drivers:")
    print(importance.head(5))

    metrics = {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc, 4),
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "feature_importance": importance.to_dict()
    }
    return metrics

# ── 5. Save ───────────────────────────────────────────────────────────────────

def save(model, metrics, feature_cols):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/attrition_xgboost.pkl")
    joblib.dump(feature_cols, "models/attrition_features.pkl")
    with open("models/attrition_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved: models/attrition_xgboost.pkl")
    log("Saved: models/attrition_metrics.json")

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics = evaluate(model, X, y, X_test, y_test, feature_cols)
    save(model, metrics, feature_cols)
    log("XGBoost attrition model complete!")