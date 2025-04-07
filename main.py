from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Load data
df = pd.read_csv("preprocess_data.csv")
df['Embedding'] = df['Embedding'].apply(eval)

model = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10

@app.post("/recommend/")
async def recommend_assessments(request: QueryRequest):
    query_emb = model.encode(request.query).reshape(1, -1)
    similarities = cosine_similarity(query_emb, df['Embedding'].tolist())[0]
    df['Similarity'] = similarities
    top_results = df.sort_values("Similarity", ascending=False).head(request.top_n)
    return {"recommendations": top_results.to_dict(orient="records")}
