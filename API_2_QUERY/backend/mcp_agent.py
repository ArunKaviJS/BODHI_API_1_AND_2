# mcp_agent.py

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from backend.embedder import get_similar_chunks
from backend.chat_logger import (
    update_chat_message,
    get_session_summary,
    update_session_summary,
    is_first_message,

)
from backend.agent_responses import (
    FaultDiagnosisResponse,
    OperationalGuidanceResponse,
    validate_llm_response,
)

# Load env vars
load_dotenv()

# === Azure OpenAI Client ===
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)
LLM_DEPLOYMENT_NAME = os.getenv("LLM_DEPLOYMENT_NAME")
MEMORY_LIMIT = 5

AGENTS = [
    "fault_diagnosis",
    "operational_guidance"
]

# === Prompts ===
INTENT_CLASSIFICATION_PROMPT = """
Classify the user's intent into one of the following categories:
- fault_diagnosis
- operational_guidance

User Query: "{query}"

Only return the intent label as a single word.
"""

INCREMENTAL_SUMMARY_PROMPT = """
You are updating a session summary that will be used for vector similarity search. 
The summary must capture the key topics, technical details, and intent in a concise but meaningful way. 
It should be cumulative, self-contained, and optimized for semantic retrieval.

Previous Summary:
{previous_summary}

Latest User Query:
{query}

Latest Assistant Response:
{response}

Task:
- Merge the new interaction into the previous summary.
- Keep it concise but information-rich.
- Preserve important entities, technical terms, and context.
- Write in natural language, not bullet points.

Return only the updated summary.
"""

FIRST_MESSAGE_SUMMARY_PROMPT = """
You are session summary that will be used for vector similarity search. 
The summary must capture the key topics, technical details, and intent in a concise but meaningful way. 
It should be cumulative, self-contained, and optimized for semantic retrieval.

User Query:
{query}

Assistant Response:
{response}

Return only the summary text.
"""

# === Utilities ===
def call_llm(prompt: str) -> str:
    """Call Azure OpenAI directly using model name."""
    response = openai_client.chat.completions.create(
        model=LLM_DEPLOYMENT_NAME,  # use model name directly
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def classify_intent(intent_input: str) -> str:
    label = call_llm(INTENT_CLASSIFICATION_PROMPT.format(query=intent_input)).lower().strip()
    return label if label in AGENTS else "unknown"


# === Agent Handlers ===
def handle_fault_diagnosis(query, chunk_context, session_summary):
    chunk_context_text = chunk_context.strip() or "(No relevant context found)"
    
    return call_llm(f"""
You are a Fault Diagnosis Agent.
Your task is to analyze issues and provide structured fault diagnosis ONLY in JSON.

IMPORTANT:
- If CONTEXT = "(No relevant context found)", then output exactly:
{{"message": "No relevant document found"}}
- Otherwise, return a JSON object with these fields:
  - issue_identified (string, required, never empty)
  - likely_causes (list of non-empty strings, required)
  - recommended_actions (list of non-empty strings, required)
  - precautionary_notes (list of non-empty strings, optional)
  - relevant_incidents (list of non-empty strings, optional)
  - source_references (list of non-empty strings, optional)

STRICT RULES:
1. Use ONLY information from QUERY, CONTEXT, and HISTORY.
2. Required fields must always be present and contain valid non-empty values.
3. Optional fields may be omitted if no valid content exists. Do NOT include empty strings, empty lists, or null values.
4. Never leave fields empty. If unknown, infer from context or mark as 'Not specified'.                    
5. Do NOT include explanations, markdown, or comments. Output raw JSON only.

INPUTS:
QUERY: "{query}"
CONTEXT: {chunk_context_text}
HISTORY: {session_summary or "No previous summary."}
""")


def handle_operational_guidance(query, chunk_context, session_summary):
    chunk = (chunk_context or "").strip() or "(No relevant context found)"
    history = session_summary or "No previous summary."
    
    return call_llm(f"""
You are an Operational Guidance Agent. Respond ONLY in valid JSON.

IMPORTANT:
- If CONTEXT = "(No relevant context found)", then output exactly:
{{"message": "No relevant document found"}}
- Otherwise, return a JSON object with these fields:
  - task (string, required, never empty)
  - tools_needed (list of non-empty strings, required)
  - step_by_step_procedure (list of 5–12 non-empty strings, each ≤25 words, required)
  - safety_checklist (list of 3–8 distinct, actionable, non-empty strings, required)

RULES:
- Use ONLY info from QUERY, CONTEXT, and HISTORY.
- "task": copy directly if mentioned, else infer a concise, meaningful title.
- "tools_needed": include only tools explicitly stated in CONTEXT or STEPS. Do not invent.
- "step_by_step_procedure": steps must come from CONTEXT. Never hallucinate.
- Never leave fields empty. If unknown, infer from context or mark as 'Not specified'.
- "safety_checklist": must contain distinct, actionable measures from CONTEXT. No duplicates, no empty entries.
- Do not include empty strings, empty lists, or null values.

STRICT OUTPUT:
- Raw JSON only.
- No schema echo, no placeholders, no extra text.

INPUTS:
QUERY: {query}
CONTEXT: {chunk}
HISTORY: {history}
""")


router_map = {
    "fault_diagnosis": handle_fault_diagnosis,
    "operational_guidance": handle_operational_guidance
}


VECTOR_SEARCH_SUMMARY_PROMPT = """
You are preparing text for vector similarity search in the domain of MACHINE OPERATION, FAULT DIAGNOSIS, and OPERATIONAL GUIDANCE.

Inputs:
- Session Summary: {session_summary}
- New Query: {query}

Task:

Step 1: Relevance Check (STRICT RULES)
- If the query contains ANY of these words or synonyms:
  ["CNC", "machine", "equipment", "control panel", "motor", "spindle", "alarm", "fault", "error", "maintenance", "startup", "shutdown", "troubleshoot", "operation", "power on", "turn on", "power off", "start", "stop"]
  → It is ALWAYS RELEVANT. Do not override this rule.
- If it clearly refers to machine use, faults, alarms, operation, or maintenance → RELEVANT.
- Otherwise → IRRELEVANT.

Step 2: Response
- If RELEVANT:
  Create a concise 1–5 sentence summary combining the new query with the session summary.
  - Preserve technical terms, machine part names, alarms, and troubleshooting details.
  - Optimize for semantic retrieval (short, precise, technical).
- If IRRELEVANT:
  Return exactly: IRRELEVANT_QUERY

IMPORTANT:
- Do not interpret beyond the rules.
- If the query matches keywords (like "How to turn on CNC machine?"), ALWAYS treat as RELEVANT.
- Output must be only the summary text OR "IRRELEVANT_QUERY".
"""



def build_vector_search_input(query: str, session_summary: str) -> str:
    """Generate optimized text for embedding-based retrieval, or detect irrelevance."""
    prompt = VECTOR_SEARCH_SUMMARY_PROMPT.format(
        session_summary=session_summary or "No previous summary.",
        query=query
    )
    vector_text = call_llm(prompt).strip()
    
    if vector_text.upper() == "IRRELEVANT_QUERY":
        return None  # Flag for irrelevant query
    return vector_text


# === Main query function (Refactored with vector search input) ===
def query_llm_with_context(user_query: str, session_id: str, machine_id: str, user_id: str, _id: str):
    """
    1. Build vector search input (query + session summary → retrieval summary)
    2. If query irrelevant → return friendly message immediately
    3. Retrieve relevant docs using retrieval summary
    4. Call correct agent with chunk context only
    5. Save chat messages once & update session summary
    """

    print(f"\n=== New Query ===")
    print(f"User Query: {user_query}")
    print(f"Session ID: {session_id}, Machine ID: {machine_id}, User ID: {user_id}, chat_id: {_id}")

    # Step 1: get session summary (for updating later)
    session_summary = get_session_summary(session_id) or ""
    print(f"\n[Session Summary]\n{session_summary if session_summary else '(No summary yet)'}")

    # Step 2: build vector search input & check relevance
    vector_search_text = build_vector_search_input(user_query, session_summary)
    if not vector_search_text:
        # Query detected as irrelevant → send immediate answer
        answer = {
            "response_type": "irrelevant_query",
            "message": "Hmm, that doesn’t seem related to machine operation. Can you clarify?"
        }
        update_chat_message(_id, answer)
        print(f"[Chat Saved] Chat ID: {_id} (Irrelevant query)")

        # Update session summary
        updated_summary = call_llm(
            INCREMENTAL_SUMMARY_PROMPT.format(
                previous_summary=session_summary,
                query=user_query,
                response=json.dumps(answer)
            )
        )
        update_session_summary(session_id, updated_summary)
        print(f"\n[Updated Summary]\n{updated_summary}")

        return answer, _id

    print(f"\n[Vector Search Input]\n{vector_search_text}")

    # Step 3: retrieve relevant chunks using vector search input
    chunks = get_similar_chunks(vector_search_text, machine_id=machine_id, top_k=3)
    print(f"\n[Chunks Retrieved] ({len(chunks)} chunks)")

    # Step 4: prepare chunk context for LLM
    chunk_context = "\n\n".join([doc for _, doc, _ in chunks]) if chunks else ""
    if not chunk_context:
        chunk_context = "(No relevant context found)"  # fallback for LLM
    for i, (_, doc, _) in enumerate(chunks, start=1):
        print(f"  Chunk {i}: {doc[:200]}...")
    print(f"Chunk Context Combined: {chunk_context[:200]}... {len(chunk_context)}")

    # Step 5: classify intent
    intent_input = f"{session_summary}\n{user_query}" if session_summary else user_query
    intent = classify_intent(intent_input)
    print(f"\n[Intent Classified] → {intent}")

    handler = router_map.get(intent)
    if not handler:
        answer = {"error": "unknown_intent", "message": "Could not determine the right agent."}
    else:
        reply_text = handler(user_query, chunk_context, session_summary)
        print("[Raw LLM Output]", reply_text)

        try:
            answer = validate_llm_response(reply_text, intent)
            print("[Reply Validated ✅] LLM response is correct and parsed as dict")
        except ValueError as e:
            # Provide fallback empty JSON matching schema
            print("[Reply Validation Failed ❌]", e)
            if intent == "fault_diagnosis":
                answer = FaultDiagnosisResponse.empty_dict()
            else:
                answer = OperationalGuidanceResponse.empty_dict()
            answer["raw"] = reply_text  # keep raw output for debugging

    # Step 6: save both user query and assistant reply once
    update_chat_message(_id, answer)
    print(f"[Chat Saved] Chat ID: {_id}")

    # Step 7: update session summary
    first_message = is_first_message(session_id)
    if first_message:
        summary_prompt = FIRST_MESSAGE_SUMMARY_PROMPT.format(
            query=user_query, response=json.dumps(answer)
        )
    else:
        summary_prompt = INCREMENTAL_SUMMARY_PROMPT.format(
            previous_summary=session_summary,
            query=user_query,
            response=json.dumps(answer)
        )

    updated_summary = call_llm(summary_prompt)

    # Update session summary directly instead of re-inserting
    update_session_summary(session_id, updated_summary)
    print(f"\n[Updated Summary]\n{updated_summary}")
    print("=== End Query ===\n")

    return answer, _id



