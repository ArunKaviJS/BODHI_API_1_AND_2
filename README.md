# BODHI_API_1_AND_2

API 1 - /title → Lambda function name: titlequeryrag  
API 2 - /query → Lambda function name: rag_mcp_chat_pipeline


API1:
Title Generation API

This is a FastAPI service that generates concise session titles from a user query using Azure OpenAI (gpt-4o) and stores them in MongoDB

---

Features

- Generates meaningful short titles (3–7 words recommended) from a user query.
- Cleans and normalizes titles to remove extra spaces, quotes, or slashes.
- Stores the generated title in a MongoDB collection `tb_sessions`.
- Returns the title along with a UTC `modifiedOn` timestamp.
- Fallback to `"Unknown Title"` if LLM cannot generate a title.

---

API2 :
AI-powered Fault Diagnosis & Operational Guidance System

 Overview

The MCP Agent is an intelligent assistant built on Azure OpenAI for handling machine operation, troubleshooting, and fault diagnosis queries.
It classifies user intent into Fault Diagnosis or Operational Guidance, retrieves relevant documentation using vector search, and provides structured JSON responses optimized for downstream systems.

This project ensures:
 Reliable intent classification
 Strictly structured JSON outputs (no free text)
Vector-based context retrieval for accurate responses
Incremental session summaries optimized for semantic search

⚙️ Features

Intent Classification
Classifies queries into:

fault_diagnosis

operational_guidance

Vector Search Integration
Uses session history + query to retrieve the most relevant context chunks.

Fault Diagnosis Agent
Returns structured JSON with:

issue_identified

likely_causes

recommended_actions

(optional) precautionary_notes, relevant_incidents, source_references

Operational Guidance Agent
Returns structured JSON with:

task

tools_needed

step_by_step_procedure

safety_checklist

Session Summarization

Maintains incremental summaries

Optimized for semantic retrieval

Irrelevant Query Handling
Detects off-topic queries and responds gracefully.

Workflow

User Query →
Input from operator/technician.

Intent Classification →
fault_diagnosis OR operational_guidance.

Vector Search Retrieval →
Context fetched from domain knowledge base.

Agent Response →
LLM outputs structured JSON.

Session Summary Update →
Maintains cumulative semantic summary for future queries.