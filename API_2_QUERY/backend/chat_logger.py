from datetime import datetime
from typing import List, Dict
from bson import ObjectId
from backend.db import chat_collection, session_collection


# ✅ Update chat message (assistant response + modified timestamp)
def update_chat_message(_id: str, answer: str):
    result = chat_collection.update_one(
        {"_id": ObjectId(_id) if ObjectId.is_valid(_id) else _id},
        {"$set": {
            "answer": answer,
            "modifiedOn": datetime.utcnow()
        }}
    )
    if result.matched_count == 0:
        print(f"⚠️ No chat found with chat_id={_id}")
    else:
        print(f"✅ Chat updated for chat_id={_id}")


# ✅ Load all sessions for a user
def load_user_chat_sessions(user_id: str) -> List[Dict]:
    return list(
        session_collection.find({"user_id": user_id}, {"_id": 0})
        .sort("created_at", -1)
    )


# ✅ Load all messages for a session
def load_chat_messages(session_id: str) -> List[Dict]:
    return list(
        chat_collection.find({"session_id": session_id}, {"_id": 0})
        .sort("timestamp", 1)
    )


# ✅ Session Summary Management
def get_session_summary(session_id: str) -> str:
    try:
        query_id = ObjectId(session_id) if ObjectId.is_valid(session_id) else session_id
        session = session_collection.find_one(
            {"_id": query_id},
            {"summarization": 1, "_id": 0}
        )
        return session.get("summarization", "") if session else ""
    except Exception:
        return ""


def update_session_summary(session_id: str, updated_summary: str):
    result = session_collection.update_one(
        {"_id": ObjectId(session_id) if ObjectId.is_valid(session_id) else session_id},
        {"$set": {
            "summarization": updated_summary,
            "modifiedOn": datetime.utcnow()
        }},
        upsert=False   # 👈 don’t create new sessions here, only update
    )
    if result.matched_count == 0:
        print(f"⚠️ No session found with session_id={session_id}")
    else:
        print(f"✅ Summary updated for session_id={session_id}")


# ✅ Check if session has messages
def is_first_message(session_id: str) -> bool:
    return chat_collection.count_documents({"session_id": session_id}) == 1




