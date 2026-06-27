from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from text_to_sql import text_to_sql_pipeline

app = FastAPI(
    title="InsightForge API",
    description="AI-Augmented Business Analytics API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load ML models once at startup ────────────────────────────────────────────

xgb_model = joblib.load("../models/attrition_xgboost.pkl")
xgb_features = joblib.load("../models/attrition_features.pkl")

# ── Request/Response models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class AttritionRequest(BaseModel):
    age: int
    gender: str
    monthly_income: float
    job_satisfaction: int
    years_at_company: int
    overtime: str
    distance_from_home: int
    education: int
    environment_satisfaction: int
    performance_rating: int
    department_name: str
    job_role: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "InsightForge API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = text_to_sql_pipeline(req.question)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/predict/attrition")
def predict_attrition(req: AttritionRequest):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    dept_map = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
    role_map = {
        "Healthcare Representative": 0, "Human Resources": 1,
        "Laboratory Technician": 2, "Manager": 3,
        "Manufacturing Director": 4, "Research Director": 5,
        "Research Scientist": 6, "Sales Executive": 7,
        "Sales Representative": 8
    }
    gender_enc = 1 if req.gender == "Male" else 0
    overtime_enc = 1 if req.overtime == "Yes" else 0
    dept_enc = dept_map.get(req.department_name, 1)
    role_enc = role_map.get(req.job_role, 6)

    input_df = pd.DataFrame([[
        req.age, gender_enc, req.monthly_income, req.job_satisfaction,
        req.years_at_company, overtime_enc, req.distance_from_home,
        req.education, req.environment_satisfaction, req.performance_rating,
        dept_enc, role_enc
    ]], columns=xgb_features)

    prob = xgb_model.predict_proba(input_df)[0][1]
    risk = "High" if prob > 0.6 else "Medium" if prob > 0.35 else "Low"

    return {
        "attrition_probability": round(float(prob), 4),
        "risk_level": risk,
        "recommendation": (
            "Immediate intervention recommended — review workload and compensation."
            if risk == "High" else
            "Monitor closely and improve engagement."
            if risk == "Medium" else
            "Employee appears stable. Maintain current engagement."
        )
    }

@app.get("/insights/summary")
def summary():
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    import os
    load_dotenv()
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        total_orders = conn.execute(text("SELECT COUNT(*) FROM fact_orders")).scalar()
        gmv = conn.execute(text("SELECT SUM(price) FROM fact_orders")).scalar()
        attrition = conn.execute(text("""
            SELECT ROUND(100.0 * SUM(CASE WHEN dal.attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1)
            FROM fact_employees fe JOIN dim_attrition_label dal ON fe.label_id = dal.label_id
        """)).scalar()
    return {
        "total_orders": total_orders,
        "gmv": round(float(gmv), 2),
        "attrition_rate_pct": float(attrition)
    }