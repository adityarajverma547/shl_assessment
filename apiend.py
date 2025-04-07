import nest_asyncio
from pyngrok import ngrok
nest_asyncio.apply()

# Step 3: Load necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Step 4: Load your data (Assume you already uploaded "preprocessed_data.csv")
df = pd.read_csv("preprocess_data.csv")
df['Embedding'] = df['Embedding'].apply(eval)  # Convert string list to actual list

# Step 5: Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 6: Define FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10

def recommend(query: str, top_n: int = 10) -> List[Dict]:
    query_emb = model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_emb, df['Embedding'].tolist())[0]
    df_temp = df.copy()
    df_temp['Similarity'] = similarities
    top_results = df_temp.sort_values('Similarity', ascending=False).head(top_n)

    return [
        {
            "Assessment Name": row['Job Solution'],
            "URL": row['Link'],
            "Remote Testing Support": row['Remote Testing'],
            "Adaptive/IRT Support": row['Adaptive/IRT'],
            "Duration": row['Duration'],
            "Test Types": row['Test Types'],
            "Similarity Score": row['Similarity']
        }
        for _, row in top_results.iterrows()
    ]

@app.post("/recommend/")
async def recommend_assessments(request: QueryRequest):
    recommendations = recommend(request.query, request.top_n)
    return {"recommendations": recommendations}

# Step 7: Run app with ngrok
public_url = ngrok.connect(8000)
print(f"🚀 Your app is running at: {public_url}/docs")

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)