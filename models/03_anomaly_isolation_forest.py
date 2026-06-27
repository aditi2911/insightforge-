import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

def log(msg): print(f"[ISOFOREST] {msg}")

# ── 1. Load orders ────────────────────────────────────────────────────────────

def load_data():
    log("Loading order data from Postgres...")
    query = """
        SELECT
            fo.order_id,
            fo.price,
            fo.freight_value,
            fo.payment_installments,
            fo.review_score,
            fo.delivery_days,
            dd.day_of_week,
            dd.month
        FROM fact_orders fo
        JOIN dim_date dd ON fo.date_id = dd.date_id
        WHERE fo.price IS NOT NULL
          AND fo.delivery_days IS NOT NULL
          AND fo.delivery_days > 0
    """
    df = pd.read_sql(query, engine)
    log(f"  → {len(df)} orders loaded")
    return df

# ── 2. Feature engineering ────────────────────────────────────────────────────

def prepare_features(df):
    log("Engineering features...")

    # encode day of week
    dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
    df["dow_num"] = df["day_of_week"].map(dow_map).fillna(0)

    # freight ratio — unusually high freight vs price is suspicious
    df["freight_ratio"] = df["freight_value"] / (df["price"] + 1)

    # fill missing review scores with median
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    feature_cols = [
        "price", "freight_value", "freight_ratio",
        "payment_installments", "review_score",
        "delivery_days", "dow_num", "month"
    ]

    X = df[feature_cols].fillna(0)
    log(f"  → {X.shape[1]} features prepared")
    return df, X, feature_cols

# ── 3. Train Isolation Forest ─────────────────────────────────────────────────

def train_model(X):
    log("Training Isolation Forest...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.05 means we expect ~5% of orders to be anomalous
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # predict: -1 = anomaly, 1 = normal
    preds = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)  # lower = more anomalous

    log(f"  → Anomalies detected: {(preds == -1).sum()} ({(preds == -1).mean()*100:.1f}%)")
    return model, scaler, preds, scores

# ── 4. Analyse anomalies ──────────────────────────────────────────────────────

def analyse(df, preds, scores):
    log("Analysing anomalies...")
    df = df.copy()
    df["anomaly"] = preds
    df["anomaly_score"] = scores
    df["is_anomaly"] = df["anomaly"] == -1

    anomalies = df[df["is_anomaly"]]

    log(f"  → Avg price (normal):   R${df[~df['is_anomaly']]['price'].mean():.2f}")
    log(f"  → Avg price (anomaly):  R${anomalies['price'].mean():.2f}")
    log(f"  → Avg delivery (normal):   {df[~df['is_anomaly']]['delivery_days'].mean():.1f} days")
    log(f"  → Avg delivery (anomaly):  {anomalies['delivery_days'].mean():.1f} days")

    # save anomaly orders
    anomalies[["order_id","price","freight_value","delivery_days",
               "review_score","anomaly_score"]].to_csv(
        "data/processed/anomaly_orders.csv", index=False)
    log(f"  → Saved {len(anomalies)} anomaly orders to data/processed/anomaly_orders.csv")

    metrics = {
        "total_orders": len(df),
        "anomaly_count": int((preds == -1).sum()),
        "anomaly_pct": round((preds == -1).mean() * 100, 2),
        "avg_price_normal": round(df[~df["is_anomaly"]]["price"].mean(), 2),
        "avg_price_anomaly": round(anomalies["price"].mean(), 2),
        "avg_delivery_normal": round(df[~df["is_anomaly"]]["delivery_days"].mean(), 1),
        "avg_delivery_anomaly": round(anomalies["delivery_days"].mean(), 1),
    }
    return metrics

# ── 5. Save ───────────────────────────────────────────────────────────────────

def save(model, scaler, feature_cols, metrics):
    joblib.dump(model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/isolation_forest_scaler.pkl")
    joblib.dump(feature_cols, "models/isolation_forest_features.pkl")
    with open("models/anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved: models/isolation_forest.pkl")
    log("Saved: models/anomaly_metrics.json")

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df, X, feature_cols = prepare_features(df)
    model, scaler, preds, scores = train_model(X)
    metrics = analyse(df, preds, scores)
    save(model, scaler, feature_cols, metrics)
    log("Isolation Forest anomaly detection complete!")