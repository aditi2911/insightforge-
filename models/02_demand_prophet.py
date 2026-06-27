import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib
import json
from prophet import Prophet

load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

def log(msg): print(f"[PROPHET] {msg}")

# ── 1. Load time series from star schema ──────────────────────────────────────

def load_data():
    log("Loading order time series from Postgres...")
    query = """
        SELECT
            dd.full_date AS ds,
            COUNT(fo.order_id) AS order_count,
            SUM(fo.price) AS revenue
        FROM fact_orders fo
        JOIN dim_date dd ON fo.date_id = dd.date_id
        WHERE fo.order_status = 'delivered'
        GROUP BY dd.full_date
        ORDER BY dd.full_date
    """
    df = pd.read_sql(query, engine)
    df["ds"] = pd.to_datetime(df["ds"])
    log(f"  → {len(df)} daily data points from {df['ds'].min().date()} to {df['ds'].max().date()}")
    return df

# ── 2. Train Prophet model ────────────────────────────────────────────────────

def train_model(df):
    log("Training Prophet forecasting model...")

    # Prophet needs columns named exactly 'ds' and 'y'
    train_df = df[["ds", "order_count"]].rename(columns={"order_count": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)
    log("  → Model trained on order volume")
    return model, train_df

# ── 3. Forecast next 90 days ──────────────────────────────────────────────────

def forecast(model, train_df):
    log("Forecasting next 90 days...")
    future = model.make_future_dataframe(periods=90)
    forecast_df = model.predict(future)

    # key columns: ds, yhat (prediction), yhat_lower, yhat_upper
    result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(90)
    result["yhat"] = result["yhat"].clip(lower=0).round(1)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0).round(1)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0).round(1)

    log(f"  → Avg predicted daily orders: {result['yhat'].mean():.1f}")
    log(f"  → Peak predicted day: {result.loc[result['yhat'].idxmax(), 'ds'].date()} ({result['yhat'].max():.0f} orders)")
    return forecast_df, result

# ── 4. Revenue model ──────────────────────────────────────────────────────────

def train_revenue_model(df):
    log("Training Prophet revenue model...")
    rev_df = df[["ds", "revenue"]].rename(columns={"revenue": "y"})
    rev_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    rev_model.fit(rev_df)
    future = rev_model.make_future_dataframe(periods=90)
    rev_forecast = rev_model.predict(future)
    log(f"  → Avg predicted daily revenue: R${rev_forecast['yhat'].tail(90).mean():.2f}")
    return rev_model, rev_forecast

# ── 5. Save ───────────────────────────────────────────────────────────────────

def save(model, rev_model, forecast_df, rev_forecast, result):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/prophet_orders.pkl")
    joblib.dump(rev_model, "models/prophet_revenue.pkl")

    # save forecast to CSV for dashboard
    forecast_df[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(
        "data/processed/orders_forecast.csv", index=False)
    rev_forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(
        "data/processed/revenue_forecast.csv", index=False)

    metrics = {
        "forecast_horizon_days": 90,
        "avg_predicted_daily_orders": round(result["yhat"].mean(), 1),
        "peak_predicted_orders": round(result["yhat"].max(), 1),
    }
    with open("models/prophet_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log("Saved: models/prophet_orders.pkl")
    log("Saved: models/prophet_revenue.pkl")
    log("Saved: data/processed/orders_forecast.csv")

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    model, train_df = train_model(df)
    forecast_df, result = forecast(model, train_df)
    rev_model, rev_forecast = train_revenue_model(df)
    save(model, rev_model, forecast_df, rev_forecast, result)
    log("Prophet forecasting complete!")