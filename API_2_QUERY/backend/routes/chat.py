from fastapi import APIRouter, HTTPException, Request
from backend.chat_logger import load_user_chat_sessions
from backend.mcp_agent import query_llm_with_context

router = APIRouter()


@router.post("/ask")
async def ask_question(request: Request):
    body = await request.json()

    session_id = body.get("sessionID") or body.get("session_id")
    machine_id = body.get("machineId") or body.get("machine_id")
    query = body.get("query")
    user_id = body.get("userId") or body.get("user_id") or "anonymous"
    _id = body.get("_id")

    if not query:
        raise HTTPException(status_code=400, detail="Please provide a question.")
    if not machine_id:
        raise HTTPException(status_code=400, detail="Please provide machine_id.")

    try:
        reply, chat_id = query_llm_with_context(query, session_id, machine_id, user_id, _id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

    return {"reply": reply, "session_id": session_id}


@router.get("/session-list/{user_id}")
def get_chat_sessions_meta(user_id: str):
    sessions = load_user_chat_sessions(user_id)
    return [
        {
            "session_id": s["session_id"],
            "session_name": s["session_name"],
            "created_at": s["created_at"]
        } for s in sessions
    ]
