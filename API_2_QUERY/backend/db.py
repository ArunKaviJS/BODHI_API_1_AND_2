# backend/db.py
import os
from pymongo import MongoClient

MONGO_VECTORS = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MONGO_BODHI = "yyyyyyyyyyyyyyyyyy"
# Main DB for chat
CHAT_DB_NAME = os.getenv("MONGO_DB_NAME")
# Separate DB for embeddings
EMBED_DB_NAME = os.getenv("MONGO_EMBED_DB_NAME", "embedding_db")
EMBED_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "manual_embeddings")

mongo_client = MongoClient(MONGO_BODHI)
mongo_embed = MongoClient(MONGO_VECTORS)

# Databases
chat_db = mongo_client[CHAT_DB_NAME]
embed_db = mongo_embed[EMBED_DB_NAME]

# Collections
chat_collection = chat_db["tb_session_chats"]
session_collection = chat_db["tb_sessions"]
embedding_collection = embed_db[EMBED_COLLECTION_NAME]