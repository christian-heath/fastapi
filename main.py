from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

model_name = "BAAI/bge-small-en"
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = SentenceTransformer(model_name)
    print(f"✅ Model {model_name} loaded")

@app.get("/health")
async def health():
    return {"status": "ready", "model": model_name, "dim": 384}

@app.post("/embed")
async def embed(request: Request):
    data = await request.json()
    text = data.get("text")
    emb = model.encode(text).tolist()
    return {"embedding": emb}

@app.post("/batch_embed")
async def batch_embed(request: Request):
    data = await request.json()
    texts = data.get("texts", [])
    embs = model.encode(texts).tolist()
    return {"embeddings": embs}
 
