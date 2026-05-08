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


# LLM
Create .env file and add your api key

GROQ_API_KEY='Your API KEY'

# Vector DB
Emebeddings created from csv data of first 500 rows
You can create the whole document embeddings by commenting out df[:500] in rag_pipline.py
Create 500 rows embeddings just to accelerate testing

