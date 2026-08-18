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

print(df.head())

#Filter to smaller dataset 
print(df['doc_name'].unique())
print(len(df['doc_name'].unique()))

unique_docs_name = df['doc_name'].unique()
unique_doc = df[df['doc_name'].isin(unique_docs_name)]
unique_doc = unique_doc.drop_duplicates(subset=['doc_name'])

print(unique_doc.head())

print(unique_doc['doc_link'])

# Download the documments from the link in the dataframe and save them as pdfs.
os.makedirs("pdfs", exist_ok=True)
for _, row in unique_doc.iterrows():
    name = row['doc_name']
    url = row['doc_link']
    try:
        response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(f"pdfs/{name}.pdf", 'wb') as f:
                f.write(response.content)
            print(f"{name} downloaded successfully.")
        else:
            print(f"Failed to download {name}. Status code: {response.status_code}")
    except Exception as e:  
        print(f"Error downloading {name}: {type(e).__name__ } : {e}")

#Create a dataset of a subset of the documents to use for testing.




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


