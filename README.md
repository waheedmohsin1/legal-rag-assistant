# legal-rag-assistant
Hierarchy-aware legal RAG system using FastAPI, LangChain, ChromaDB, and Groq LLMs.

# Features
- Hierarchical legal chunking
- Semantic search
- Metadata-aware retrieval
- Chroma vector database
- FastAPI backend
- Interactive legal assistant UI
- Citation generation

# Architecture
CSV Legal Documents
    ↓
Hierarchy Grouping
    ↓
Parent Documents
    ↓
Conditional Child Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
LLM Response Generation

# Setup
uv add -r requirements.txt 

run python app.py
