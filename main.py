# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your RAG pipeline function
from retrieval import rag_pipeline  # make sure this is your function in retrieval.py

app = FastAPI()

# Allow your React frontend to access this backend
origins = ["http://localhost:5173"]  # Vite dev server

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class QueryRequest(BaseModel):
    question: str

# POST endpoint to get RAG response
@app.post("/chat")
async def chat(request: QueryRequest):
    answer = rag_pipeline(request.question)
    return {"answer": answer}
