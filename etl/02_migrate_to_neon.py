import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

LOCAL_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
NEON_URL = "postgresql://neondb_owner:npg_ypPfVCMiRu79@ep-mute-shadow-at1oedtq.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require"

local = create_engine(LOCAL_URL)
neon  = create_engine(NEON_URL)

TABLES = [
    "dim_date",
    "dim_customer",
    "dim_product",
    "dim_geography",
    "dim_department",
    "dim_attrition_label",
    "fact_orders",
    "fact_employees",
]

def migrate():
    for table in TABLES:
        print(f"Migrating {table}...")
        df = pd.read_sql(f"SELECT * FROM {table}", local)
        with neon.begin() as conn:
            conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
        df.to_sql(table, neon, if_exists="append", index=False, method="multi", chunksize=500)
        print(f"  → {len(df)} rows migrated")
    print("Migration complete!")

if __name__ == "__main__":
    migrate()