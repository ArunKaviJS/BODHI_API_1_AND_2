from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.chat import router as chat_router
 # optional

app = FastAPI(
    title="RAG System with MCP + Agentic AI",
    description="Retrieval-Augmented Generation with OpenAI, ChromaDB, MCP, Agentic AI, and FastAPI backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"status": "ok", "message": "API is live"}