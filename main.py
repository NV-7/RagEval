from fastapi import FastAPI
import requests
import psycopg
import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

ds = load_dataset("PatronusAI/financebench")
df = ds["train"].to_pandas()

print(df.columns.tolist())
print(len(df['company'].unique().tolist()))


app = FastAPI()
@app.get("/health")
def health_check():
    try:
        conn = psycopg.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        conn.close()
        return {"status": "healthy", "database_connected": True, "pgvector_installed": result is not None}
    except Exception as e:
        return {"status": "unhealthy", "database_connected": False, "error": str(e)}


