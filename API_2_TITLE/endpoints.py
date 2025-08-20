import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from bson import ObjectId
from openai import AzureOpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone

# Load .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# ✅ Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY")
)

# ✅ MongoDB Connection
MONGO_URI = os.getenv("bodhimonggo")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["bodhi-dev"]
tbl_session_chat = db["tb_sessions"]  # collection name


# ✅ Request schema (treat _id as str, not ObjectId)
class QuerySchema(BaseModel):
    id: str = Field(alias="_id")  # input comes as `_id`
    query: str

    @field_validator("id", mode="before")
    def validate_object_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId: {v}")
        return str(v)  # keep it as str in Pydantic

    class Config:
        populate_by_name = True
        extra = "forbid"


# ✅ Response schema with validation
class TitleResponse(BaseModel):
    title: str
    modifiedOn: str

    # Ensure title is never empty
    @model_validator(mode="after")
    def validate_title(self):
        if not self.title or not self.title.strip():
            raise ValueError("Generated title cannot be empty")
        return self


# ✅ Clean LLM output
def clean_title(raw_title: str) -> str:
    cleaned = raw_title.strip().strip('"').strip("'")
    cleaned = re.sub(r"[\"'\\\\]", "", cleaned)  # remove quotes & slashes
    cleaned = re.sub(r"\s+", " ", cleaned)       # normalize spaces
    return cleaned.strip()


# ✅ Function to generate short title using Azure OpenAI
def generate_title_with_llm(query: str) -> str:
    prompt = f"""
You are a helpful assistant.
The user will provide a query or text.
Your task is to create a clear, meaningful session title.

⚠️ Important:
- Do NOT leave the response empty.
- If you cannot find a meaningful title, return exactly: Unknown Title
- Otherwise, generate a concise title (3–7 words preferred, longer allowed if necessary).

Query: {query}
Title:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a concise title generator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=40,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()

    # Fallback if model still returns empty
    if not raw or raw.lower() in ["", "none", "null"]:
        return "Unknown Title"

    return raw


# ✅ Register routes
def register_routes(app: FastAPI):

    @app.post("/title", response_model=TitleResponse)
    async def title(payload: QuerySchema):
        try:
            query_text = payload.query.strip()
            if not query_text:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Generate + clean
            llm_title = generate_title_with_llm(query_text)
            final_title = clean_title(llm_title) or "Unknown Title"
            modified_on = datetime.now(timezone.utc)

            # ✅ Convert string → ObjectId before DB update
            mongo_id = ObjectId(payload.id)

            update_result = tbl_session_chat.update_one(
                {"_id": mongo_id},
                {"$set": {
                    "title": final_title,
                    "modifiedOn": modified_on
                }}
            )

            if update_result.matched_count == 0:
                raise HTTPException(status_code=404, detail=f"_id {payload.id} not found in collection")

            return TitleResponse(
                title=final_title,
                modifiedOn=modified_on.isoformat()
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ✅ Register routes when running app
register_routes(app)
