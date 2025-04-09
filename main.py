from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# Load data
df = pd.read_csv("preprocess_data.csv")
df['Embedding'] = df['Embedding'].apply(eval)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# App init
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Recommendation endpoint
@app.post("/recommend")
async def recommend_assessments(request: QueryRequest):
    query_emb = model.encode(request.query).reshape(1, -1)
    similarities = cosine_similarity(query_emb, df['Embedding'].tolist())[0]
    df['Similarity'] = similarities
    top_results = df.sort_values("Similarity", ascending=False).head(10)

    response = []
    for _, row in top_results.iterrows():
        response.append({
            "url": row["Link"],
            "adaptive_support": row["Adaptive/IRT"],
            "description": row["Job Solution"],
            "duration": int(row["Duration"]),
            "remote_support": row["Remote Testing"],
            "test_type": row["Test Types"].split(",") if isinstance(row["Test Types"], str) else []
        })

    return {"recommended_assessments": response}
