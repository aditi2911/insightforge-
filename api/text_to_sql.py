import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.3-70b-versatile"

SCHEMA = """
You are a PostgreSQL expert. Generate ONLY valid SQL queries for this schema:

TABLES:
- fact_orders(order_id, customer_id, product_id, date_id, geo_id, order_status, payment_type, payment_installments, price, freight_value, review_score, delivery_days)
- fact_employees(employee_id, department_id, date_id, label_id, age, gender, monthly_income, job_satisfaction, years_at_company, overtime, distance_from_home, education, environment_satisfaction, performance_rating)
- dim_date(date_id, full_date, year, quarter, month, month_name, week, day_of_week)
- dim_customer(customer_id, city, state, rfm_segment)
- dim_product(product_id, category_name, product_name, weight_kg)
- dim_department(department_id, department_name, job_role)
- dim_geography(geo_id, city, state, region)
- dim_attrition_label(label_id, attrition, risk_tier)

RULES:
1. Always use table aliases
2. Always JOIN correctly using foreign keys
3. LIMIT results to 20 rows maximum
4. For revenue/price questions, use SUM(fo.price)
5. For attrition questions, join dim_attrition_label and filter on attrition='Yes'
6. Return ONLY the SQL query, no explanation, no markdown, no backticks
"""

BLOCKED = ["drop", "delete", "insert", "update", "alter", "truncate", "create", "grant"]

def is_safe(sql: str) -> bool:
    return not any(word in sql.lower() for word in BLOCKED)

def generate_sql(question: str) -> str:
    prompt = f"{SCHEMA}\n\nUser question: {question}\n\nSQL query:"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql

def execute_sql(sql: str) -> pd.DataFrame:
    if not is_safe(sql):
        raise ValueError("Unsafe SQL detected — query blocked.")
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def generate_narrative(question: str, sql: str, df: pd.DataFrame) -> str:
    data_summary = df.to_string(index=False) if len(df) <= 10 else df.head(10).to_string(index=False)
    prompt = f"""You are a senior business analyst. Given:

User Question: {question}
Query Results:
{data_summary}

Write a concise 2-3 sentence business insight in plain English.
Focus on the key finding, business impact, and one actionable recommendation.
Do not mention SQL or technical details."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def text_to_sql_pipeline(question: str) -> dict:
    try:
        sql = generate_sql(question)
        df = execute_sql(sql)
        narrative = generate_narrative(question, sql, df)
        return {
            "question": question,
            "sql": sql,
            "results": df.to_dict(orient="records"),
            "narrative": narrative,
            "row_count": len(df),
            "error": None
        }
    except Exception as e:
        return {
            "question": question,
            "sql": "",
            "results": [],
            "narrative": "",
            "row_count": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    questions = [
        "What are the top 5 product categories by total revenue?",
        "Which department has the highest attrition rate?",
        "What is the average delivery time by state?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = text_to_sql_pipeline(q)
        print(f"SQL: {result['sql']}")
        print(f"Rows: {result['row_count']}")
        print(f"Narrative: {result['narrative']}")
        if result['error']:
            print(f"ERROR: {result['error']}")