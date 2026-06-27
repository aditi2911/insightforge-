import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib
import json

load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

st.set_page_config(
    page_title="InsightForge",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DB connection (cached) ────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data(ttl=300)
def query(sql):
    return pd.read_sql(sql, get_engine())

# ── Load models ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model    = joblib.load("models/attrition_xgboost.pkl")
    features = joblib.load("models/attrition_features.pkl")
    return model, features

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("InsightForge")
st.sidebar.caption("AI-Augmented Business Analytics")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🛒 E-Commerce Analytics",
    "👥 HR & Attrition",
    "🔮 Demand Forecast",
    "🚨 Anomaly Detection",
    "🤖 Attrition Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🔥 InsightForge — Business Intelligence Platform")
    st.caption("Powered by PostgreSQL · XGBoost · Prophet · Isolation Forest · Gemini AI")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_orders = query("SELECT COUNT(*) as n FROM fact_orders").iloc[0,0]
    gmv = query("SELECT SUM(price) as g FROM fact_orders").iloc[0,0]
    attrition_rate = query("""
        SELECT ROUND(100.0 * SUM(CASE WHEN dal.attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM fact_employees fe JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
    """).iloc[0,0]
    anomaly_count = query("""
        SELECT COUNT(*) FROM fact_orders fo
        JOIN dim_date dd ON fo.date_id = dd.date_id
        WHERE fo.price > 500 AND fo.delivery_days > 20
    """).iloc[0,0]
    avg_review = query("SELECT ROUND(AVG(review_score),2) FROM fact_orders WHERE review_score IS NOT NULL").iloc[0,0]

    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("GMV", f"R${gmv:,.0f}")
    col3.metric("Attrition Rate", f"{attrition_rate}%")
    col4.metric("Flagged Orders", f"{anomaly_count:,}")
    col5.metric("Avg Review", f"⭐ {avg_review}")

    st.divider()

    # Revenue trend
    st.subheader("Revenue Trend")
    rev = query("""
        SELECT dd.year, dd.month, SUM(fo.price) as revenue
        FROM fact_orders fo JOIN dim_date dd ON fo.date_id = dd.date_id
        WHERE fo.order_status = 'delivered'
        GROUP BY dd.year, dd.month ORDER BY dd.year, dd.month
    """)
    rev["period"] = rev["year"].astype(str) + "-" + rev["month"].astype(str).str.zfill(2)
    fig = px.area(rev, x="period", y="revenue", title="Monthly Revenue (R$)",
                  color_discrete_sequence=["#FF6B6B"])
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (R$)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Two columns
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Orders by Status")
        status = query("SELECT order_status, COUNT(*) as n FROM fact_orders GROUP BY order_status ORDER BY n DESC")
        fig2 = px.pie(status, names="order_status", values="n", hole=0.4,
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Attrition by Department")
        dept = query("""
            SELECT dd.department_name,
                   ROUND(100.0 * SUM(CASE WHEN dal.attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
            FROM fact_employees fe
            JOIN dim_department dd ON fe.department_id = dd.department_id
            JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
            GROUP BY dd.department_name
        """)
        fig3 = px.bar(dept, x="department_name", y="rate",
                      color="rate", color_continuous_scale="Reds",
                      labels={"rate": "Attrition %"})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — E-COMMERCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛒 E-Commerce Analytics":
    st.title("🛒 E-Commerce Analytics")

    col1, col2, col3 = st.columns(3)
    gmv = query("SELECT SUM(price) FROM fact_orders").iloc[0,0]
    avg_order = query("SELECT AVG(price) FROM fact_orders").iloc[0,0]
    delay_rate = query("""
        SELECT ROUND(100.0*SUM(CASE WHEN delivery_days > 30 THEN 1 ELSE 0 END)/COUNT(*),1)
        FROM fact_orders WHERE delivery_days IS NOT NULL
    """).iloc[0,0]
    col1.metric("Total GMV", f"R${gmv:,.0f}")
    col2.metric("Avg Order Value", f"R${avg_order:.2f}")
    col3.metric("Delay Rate (>30d)", f"{delay_rate}%")

    st.divider()

    # Top categories
    st.subheader("Revenue by Product Category")
    cat = query("""
        SELECT dp.category_name, SUM(fo.price) as revenue, COUNT(fo.order_id) as orders
        FROM fact_orders fo JOIN dim_product dp ON fo.product_id = dp.product_id
        WHERE dp.category_name IS NOT NULL
        GROUP BY dp.category_name ORDER BY revenue DESC LIMIT 15
    """)
    fig = px.bar(cat, x="revenue", y="category_name", orientation="h",
                 color="revenue", color_continuous_scale="Blues",
                 labels={"revenue":"Revenue (R$)","category_name":"Category"})
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Payment types
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Payment Methods")
        pay = query("SELECT payment_type, COUNT(*) as n FROM fact_orders GROUP BY payment_type ORDER BY n DESC")
        fig2 = px.pie(pay, names="payment_type", values="n", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Delivery Time Distribution")
        delivery = query("SELECT delivery_days FROM fact_orders WHERE delivery_days BETWEEN 1 AND 60")
        fig3 = px.histogram(delivery, x="delivery_days", nbins=30,
                            color_discrete_sequence=["#4ECDC4"],
                            labels={"delivery_days":"Delivery Days"})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # State-level revenue
    st.subheader("Revenue by State")
    state = query("""
        SELECT dg.state, SUM(fo.price) as revenue
        FROM fact_orders fo JOIN dim_geography dg ON fo.geo_id = dg.geo_id
        GROUP BY dg.state ORDER BY revenue DESC
    """)
    fig4 = px.bar(state, x="state", y="revenue",
                  color="revenue", color_continuous_scale="Viridis")
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HR & ATTRITION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥 HR & Attrition":
    st.title("👥 HR & Attrition Analytics")

    col1, col2, col3, col4 = st.columns(4)
    total_emp = query("SELECT COUNT(*) FROM fact_employees").iloc[0,0]
    attr_count = query("""
        SELECT COUNT(*) FROM fact_employees fe
        JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
        WHERE dal.attrition = 'Yes'
    """).iloc[0,0]
    avg_income = query("SELECT AVG(monthly_income) FROM fact_employees").iloc[0,0]
    avg_tenure = query("SELECT AVG(years_at_company) FROM fact_employees").iloc[0,0]

    col1.metric("Total Employees", f"{total_emp:,}")
    col2.metric("Attrition Count", f"{attr_count}")
    col3.metric("Avg Monthly Income", f"R${avg_income:,.0f}")
    col4.metric("Avg Tenure", f"{avg_tenure:.1f} yrs")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Attrition by Overtime")
        ot = query("""
            SELECT fe.overtime,
                   SUM(CASE WHEN dal.attrition='Yes' THEN 1 ELSE 0 END) as left_co,
                   SUM(CASE WHEN dal.attrition='No' THEN 1 ELSE 0 END) as stayed
            FROM fact_employees fe
            JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
            GROUP BY fe.overtime
        """)
        fig = px.bar(ot, x="overtime", y=["left_co","stayed"],
                     barmode="group", labels={"value":"Employees","overtime":"Overtime"},
                     color_discrete_sequence=["#FF6B6B","#4ECDC4"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Income vs Attrition")
        inc = query("""
            SELECT fe.monthly_income, dal.attrition
            FROM fact_employees fe
            JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
        """)
        fig2 = px.box(inc, x="attrition", y="monthly_income",
                      color="attrition",
                      color_discrete_map={"Yes":"#FF6B6B","No":"#4ECDC4"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Satisfaction vs Attrition")
    sat = query("""
        SELECT fe.job_satisfaction, fe.environment_satisfaction, dal.attrition
        FROM fact_employees fe
        JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
    """)
    fig3 = px.scatter(sat, x="job_satisfaction", y="environment_satisfaction",
                      color="attrition",
                      color_discrete_map={"Yes":"#FF6B6B","No":"#4ECDC4"},
                      opacity=0.6, title="Job vs Environment Satisfaction")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecasting — Prophet Model")

    try:
        forecast = pd.read_csv("data/processed/orders_forecast.csv", parse_dates=["ds"])
        rev_forecast = pd.read_csv("data/processed/revenue_forecast.csv", parse_dates=["ds"])

        col1, col2, col3 = st.columns(3)
        future = forecast[forecast["ds"] > forecast["ds"].max() - pd.Timedelta(days=90)]
        col1.metric("Avg Daily Orders (forecast)", f"{future['yhat'].mean():.0f}")
        col2.metric("Peak Day", str(future.loc[future['yhat'].idxmax(),'ds'].date()))
        col3.metric("Peak Orders", f"{future['yhat'].max():.0f}")

        st.subheader("Order Volume Forecast (Next 90 Days)")
        fig = go.Figure([
            go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                       name="Forecast", line=dict(color="#FF6B6B")),
            go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                       name="Upper Bound", line=dict(dash="dot", color="#FFA07A")),
            go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                       name="Lower Bound", line=dict(dash="dot", color="#FFA07A"),
                       fill="tonexty", fillcolor="rgba(255,160,122,0.15)"),
        ])
        fig.update_layout(xaxis_title="Date", yaxis_title="Orders", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue Forecast")
        fig2 = go.Figure([
            go.Scatter(x=rev_forecast["ds"], y=rev_forecast["yhat"],
                       name="Revenue Forecast", line=dict(color="#4ECDC4")),
            go.Scatter(x=rev_forecast["ds"], y=rev_forecast["yhat_upper"],
                       fill="tonexty", fillcolor="rgba(78,205,196,0.15)",
                       name="Upper", line=dict(dash="dot",color="#4ECDC4")),
        ])
        fig2.update_layout(xaxis_title="Date", yaxis_title="Revenue (R$)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    except FileNotFoundError:
        st.error("Forecast files not found. Run models/02_demand_prophet.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Anomaly Detection":
    st.title("🚨 Order Anomaly Detection — Isolation Forest")

    try:
        anomalies = pd.read_csv("data/processed/anomaly_orders.csv")

        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalies Detected", f"{len(anomalies):,}")
        col2.metric("Avg Anomaly Price", f"R${anomalies['price'].mean():.2f}")
        col3.metric("Avg Anomaly Delivery", f"{anomalies['delivery_days'].mean():.1f} days")

        st.divider()

        st.subheader("Anomalous Orders — Price vs Delivery Days")
        fig = px.scatter(anomalies, x="delivery_days", y="price",
                         color="anomaly_score",
                         color_continuous_scale="Reds_r",
                         hover_data=["order_id","review_score"],
                         labels={"delivery_days":"Delivery Days","price":"Price (R$)"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 20 Most Anomalous Orders")
        top = anomalies.nsmallest(20, "anomaly_score")[
            ["order_id","price","freight_value","delivery_days","review_score","anomaly_score"]
        ].reset_index(drop=True)
        st.dataframe(top, use_container_width=True)

    except FileNotFoundError:
        st.error("Anomaly file not found. Run models/03_anomaly_isolation_forest.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ATTRITION PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Attrition Predictor":
    st.title("🤖 Employee Attrition Risk Predictor")
    st.caption("Enter employee details to get an AI-powered attrition risk score")

    model, features = load_models()

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 60, 35)
        monthly_income = st.number_input("Monthly Income (R$)", 1000, 20000, 5000)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        years_at_company = st.slider("Years at Company", 0, 40, 5)

    with col2:
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
        education = st.slider("Education Level (1-5)", 1, 5, 3)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)

    with col3:
        performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist",
                                              "Laboratory Technician", "Manufacturing Director",
                                              "Healthcare Representative", "Manager",
                                              "Sales Representative", "Research Director",
                                              "Human Resources"])

    if st.button("🔮 Predict Attrition Risk", type="primary"):
        # encode inputs same way as training
        gender_enc = 1 if gender == "Male" else 0
        overtime_enc = 1 if overtime == "Yes" else 0
        dept_map = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
        role_map = {"Healthcare Representative":0,"Human Resources":1,"Laboratory Technician":2,
                    "Manager":3,"Manufacturing Director":4,"Research Director":5,
                    "Research Scientist":6,"Sales Executive":7,"Sales Representative":8}
        dept_enc = dept_map.get(department, 1)
        role_enc = role_map.get(job_role, 6)

        input_data = pd.DataFrame([[
            age, gender_enc, monthly_income, job_satisfaction,
            years_at_company, overtime_enc, distance_from_home,
            education, environment_satisfaction, performance_rating,
            dept_enc, role_enc
        ]], columns=features)

        prob = model.predict_proba(input_data)[0][1]
        risk = "🔴 High Risk" if prob > 0.6 else "🟡 Medium Risk" if prob > 0.35 else "🟢 Low Risk"

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Attrition Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", risk)
        with col_b:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Attrition Risk %"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"darkred"},
                       "steps":[
                           {"range":[0,35],"color":"#4ECDC4"},
                           {"range":[35,60],"color":"#FFE66D"},
                           {"range":[60,100],"color":"#FF6B6B"}
                       ]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        if overtime == "Yes" and prob > 0.5:
            st.warning("⚠️ Overtime is the strongest attrition driver. Consider workload rebalancing.")
        if monthly_income < 3000 and prob > 0.4:
            st.warning("⚠️ Below-average income combined with other factors increases risk significantly.")