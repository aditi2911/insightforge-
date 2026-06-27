# 🔥 InsightForge — AI-Augmented Business Analytics Platform

> Predictive analytics + GenAI Text-to-SQL platform built on real e-commerce and HR datasets

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fgrimap7xeqhleypnnkhce.streamlit.app)

## 🚀 Live Demo
- **Dashboard**: https://fgrimap7xeqhleypnnkhce.streamlit.app
- **API Docs**: (Render URL — coming soon)

## 📊 What It Does

InsightForge combines a PostgreSQL data warehouse, three ML models, and a GenAI Text-to-SQL layer into a single business intelligence platform.

**Ask it anything:**
> *"Which product category had the highest revenue in Q4 2017?"*
> *"What department has the highest attrition rate?"*

It generates SQL, runs it, and returns a business narrative — powered by Llama 3.

## 🏗️ Architecture
Raw Data (CSV) → ETL Pipeline → PostgreSQL Star Schema (Neon)

↓

┌─────────────────────────┐

│   ML Models              │

│   • XGBoost (attrition) │

│   • Prophet (forecast)   │

│   • Isolation Forest     │

└─────────────────────────┘

↓

┌───────────────────────────────────┐

│  FastAPI Backend                   │

│  • /query (Text-to-SQL)           │

│  • /predict/attrition             │

│  • /insights/summary              │

└───────────────────────────────────┘

↓

Streamlit Dashboard (7 pages)

## 📁 Project Structure
insightforge/

├── etl/                    # ETL pipeline scripts

│   ├── 01_etl_pipeline.py  # Load CSVs → PostgreSQL star schema

│   └── 02_migrate_to_neon.py

├── models/                 # ML model training scripts

│   ├── 01_attrition_xgboost.py

│   ├── 02_demand_prophet.py

│   └── 03_anomaly_isolation_forest.py

├── dashboard/              # Streamlit app

│   └── app.py              # 7-page dashboard

├── api/                    # FastAPI backend

│   ├── main.py

│   └── text_to_sql.py

├── sql/                    # Schema definitions

│   └── 01_create_schema.sql

└── data/

├── raw/                # Source CSVs (gitignored)

└── processed/          # Model outputs

## 🤖 ML Models

| Model | Task | Performance |
|-------|------|-------------|
| XGBoost | Employee attrition prediction | ROC-AUC: 0.77 |
| Prophet | Order demand forecasting | 90-day horizon |
| Isolation Forest | Order anomaly detection | 4,824 anomalies flagged (5%) |

## 📦 Datasets

- **ShopPulse**: 99,441 orders from Kaggle Olist Brazilian E-Commerce dataset
- **TalentLens**: 1,470 employee records from IBM HR Analytics dataset

## 🗄️ Star Schema

- `fact_orders` — 99K order transactions
- `fact_employees` — 1,470 HR records
- `dim_date`, `dim_customer`, `dim_product`, `dim_geography`, `dim_department`, `dim_attrition_label`

## 🔑 Key Findings

- **Overtime** is the #1 attrition driver (5.19x class imbalance handled)
- **Health & Beauty** is the top revenue category (R$1.25M)
- **Black Friday 2018** predicted as peak demand day
- **4,824 orders** flagged as anomalous — avg price 4.3x higher than normal

## 🛠️ Tech Stack

`Python` `PostgreSQL` `SQLAlchemy` `Pandas` `XGBoost` `Prophet` `Scikit-learn` `Streamlit` `Plotly` `FastAPI` `Groq/Llama-3` `LangChain` `Neon` `Render` `Docker`

## ⚙️ Run Locally

```bash
# 1. Clone
git clone https://github.com/aditi2911/insightforge-.git
cd insightforge-

# 2. Install
pip install -r requirements.txt

# 3. Set up .env
cp .env.example .env
# Fill in DB credentials and GROQ_API_KEY

# 4. Run ETL
python etl/01_etl_pipeline.py

# 5. Train models
python models/01_attrition_xgboost.py
python models/02_demand_prophet.py
python models/03_anomaly_isolation_forest.py

# 6. Launch dashboard
streamlit run dashboard/app.py

# 7. Launch API
cd api && uvicorn main:app --reload
```

## 👩‍💻 Author

**Aditi** — BCA Graduate | HR → Tech Transition | Data Analytics & AI/ML
- Portfolio: [aditiport.vercel.app](https://aditiport.vercel.app)
- GitHub: [github.com/aditi2911](https://github.com/aditi2911)