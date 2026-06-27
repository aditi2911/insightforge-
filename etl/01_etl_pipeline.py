import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

RAW = "data/raw"

# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg): print(f"[ETL] {msg}")

def upsert(df, table, engine, conflict_col=None):
    """Load df into table. Uses to_sql with replace-safe approach."""
    df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    log(f"  → {len(df)} rows loaded into {table}")

# ── 1. dim_date ──────────────────────────────────────────────────────────────

def load_dim_date():
    log("Building dim_date...")
    orders = pd.read_csv(f"{RAW}/olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp"])
    dates = orders["order_purchase_timestamp"].dropna().dt.date
    dates = pd.to_datetime(pd.Series(dates.unique()))

    df = pd.DataFrame({
        "full_date":   dates.dt.date,
        "year":        dates.dt.year,
        "quarter":     dates.dt.quarter,
        "month":       dates.dt.month,
        "month_name":  dates.dt.strftime("%B"),
        "week":        dates.dt.isocalendar().week.astype(int),
        "day_of_week": dates.dt.strftime("%A"),
    }).drop_duplicates("full_date").sort_values("full_date")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_date RESTART IDENTITY CASCADE"))
    upsert(df, "dim_date", engine)

# ── 2. dim_customer ───────────────────────────────────────────────────────────

def load_dim_customer():
    log("Building dim_customer...")
    df = pd.read_csv(f"{RAW}/olist_customers_dataset.csv")
    df = df.rename(columns={
        "customer_id":   "customer_id",
        "customer_city": "city",
        "customer_state":"state",
    })[["customer_id","city","state"]].drop_duplicates("customer_id")
    df["rfm_segment"] = None   # will be filled after RFM analysis in Week 2

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_customer RESTART IDENTITY CASCADE"))
    upsert(df, "dim_customer", engine)

# ── 3. dim_product ────────────────────────────────────────────────────────────

def load_dim_product():
    log("Building dim_product...")
    products = pd.read_csv(f"{RAW}/olist_products_dataset.csv")
    translation = pd.read_csv(f"{RAW}/product_category_name_translation.csv")
    df = products.merge(translation, on="product_category_name", how="left")
    df = df.rename(columns={
        "product_id":                    "product_id",
        "product_category_name_english": "category_name",
        "product_weight_g":              "weight_kg",
    })
    df["product_name"] = df["category_name"]   # Olist has no individual product names
    df["weight_kg"] = (df["weight_kg"] / 1000).round(3)
    df = df[["product_id","category_name","product_name","weight_kg"]].drop_duplicates("product_id")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_product RESTART IDENTITY CASCADE"))
    upsert(df, "dim_product", engine)

# ── 4. dim_geography ──────────────────────────────────────────────────────────

def load_dim_geography():
    log("Building dim_geography...")
    customers = pd.read_csv(f"{RAW}/olist_customers_dataset.csv")
    df = customers[["customer_city","customer_state"]].drop_duplicates()
    df = df.rename(columns={"customer_city":"city","customer_state":"state"})
    df["region"] = df["state"].map({
        "SP":"Southeast","RJ":"Southeast","MG":"Southeast","ES":"Southeast",
        "PR":"South","SC":"South","RS":"South",
        "MT":"Center-West","MS":"Center-West","GO":"Center-West","DF":"Center-West",
        "BA":"Northeast","PE":"Northeast","CE":"Northeast","MA":"Northeast",
        "PB":"Northeast","RN":"Northeast","AL":"Northeast","SE":"Northeast","PI":"Northeast",
        "PA":"North","AM":"North","RO":"North","AC":"North","AP":"North","RR":"North","TO":"North",
    }).fillna("Other")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_geography RESTART IDENTITY CASCADE"))
    upsert(df, "dim_geography", engine)

# ── 5. dim_department ─────────────────────────────────────────────────────────

def load_dim_department():
    log("Building dim_department...")
    hr = pd.read_csv(f"{RAW}/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = hr[["Department","JobRole"]].drop_duplicates()
    df = df.rename(columns={"Department":"department_name","JobRole":"job_role"})

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_department RESTART IDENTITY CASCADE"))
    upsert(df, "dim_department", engine)

# ── 6. dim_attrition_label ────────────────────────────────────────────────────

def load_dim_attrition_label():
    log("Building dim_attrition_label...")
    df = pd.DataFrame({
        "attrition": ["Yes","Yes","Yes","No","No","No"],
        "risk_tier": ["High","Medium","Low","High","Medium","Low"],
    })
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE dim_attrition_label RESTART IDENTITY CASCADE"))
    upsert(df, "dim_attrition_label", engine)

# ── 7. fact_orders ────────────────────────────────────────────────────────────

def load_fact_orders():
    log("Building fact_orders...")
    orders   = pd.read_csv(f"{RAW}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"])
    items    = pd.read_csv(f"{RAW}/olist_order_items_dataset.csv")
    payments = pd.read_csv(f"{RAW}/olist_order_payments_dataset.csv")
    reviews  = pd.read_csv(f"{RAW}/olist_order_reviews_dataset.csv")
    customers= pd.read_csv(f"{RAW}/olist_customers_dataset.csv")

    # aggregate items per order (take first item's product, sum price+freight)
    items_agg = items.groupby("order_id").agg(
        product_id=("product_id","first"),
        price=("price","sum"),
        freight_value=("freight_value","sum")
    ).reset_index()

    # aggregate payments
    pay_agg = payments.groupby("order_id").agg(
        payment_type=("payment_type","first"),
        payment_installments=("payment_installments","max")
    ).reset_index()

    # best review per order
    rev_agg = reviews.groupby("order_id")["review_score"].mean().round(0).astype("Int64").reset_index()

    # delivery days
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days

    # date lookup
    date_map = pd.read_sql("SELECT date_id, full_date FROM dim_date", engine)
    date_map["full_date"] = pd.to_datetime(date_map["full_date"]).dt.date
    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.date

    # geo lookup
    geo_map = pd.read_sql("SELECT geo_id, city, state FROM dim_geography", engine)
    cust_geo = customers.merge(geo_map,
        left_on=["customer_city","customer_state"],
        right_on=["city","state"], how="left")[["customer_id","geo_id"]]

    # assemble
    df = (orders
        .merge(items_agg,  on="order_id", how="left")
        .merge(pay_agg,    on="order_id", how="left")
        .merge(rev_agg,    on="order_id", how="left")
        .merge(date_map,   left_on="purchase_date", right_on="full_date", how="left")
        .merge(cust_geo,   on="customer_id", how="left")
    )

    df = df[[
        "order_id","customer_id","product_id","date_id","geo_id",
        "order_status","payment_type","payment_installments",
        "price","freight_value","review_score","delivery_days"
    ]].drop_duplicates("order_id")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE fact_orders RESTART IDENTITY CASCADE"))
    upsert(df, "fact_orders", engine)

# ── 8. fact_employees ─────────────────────────────────────────────────────────

def load_fact_employees():
    log("Building fact_employees...")
    hr = pd.read_csv(f"{RAW}/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    dept_map  = pd.read_sql("SELECT department_id, department_name, job_role FROM dim_department", engine)
    label_map = pd.read_sql("SELECT label_id, attrition FROM dim_attrition_label WHERE risk_tier='Medium'", engine)

    # get a fixed date_id (use earliest date as placeholder — employees aren't time-series yet)
    date_id = pd.read_sql("SELECT date_id FROM dim_date ORDER BY full_date LIMIT 1", engine).iloc[0,0]

    df = hr.merge(dept_map, left_on=["Department","JobRole"], right_on=["department_name","job_role"], how="left")
    df = df.merge(label_map, left_on="Attrition", right_on="attrition", how="left")

    df["date_id"] = date_id

    df = df.rename(columns={
        "Age":"age","Gender":"gender","MonthlyIncome":"monthly_income",
        "JobSatisfaction":"job_satisfaction","YearsAtCompany":"years_at_company",
        "OverTime":"overtime","DistanceFromHome":"distance_from_home",
        "Education":"education","EnvironmentSatisfaction":"environment_satisfaction",
        "PerformanceRating":"performance_rating",
    })[[
        "department_id","date_id","label_id","age","gender","monthly_income",
        "job_satisfaction","years_at_company","overtime","distance_from_home",
        "education","environment_satisfaction","performance_rating"
    ]]

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE fact_employees RESTART IDENTITY CASCADE"))
    upsert(df, "fact_employees", engine)

# ── RUN ALL ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("Starting InsightForge ETL...")
    load_dim_date()
    load_dim_customer()
    load_dim_product()
    load_dim_geography()
    load_dim_department()
    load_dim_attrition_label()
    load_fact_orders()
    load_fact_employees()
    log("ETL complete!")