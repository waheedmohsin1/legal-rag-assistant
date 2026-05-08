import pandas as pd
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/top10_combined.csv")
print('Data Loaded')
# Testing only
df = df[:500]

# =========================================================
# CLEAN DATA
# =========================================================

unused_cols = [
    'Unnamed: 9',
    'Unnamed: 10',
    'Unnamed: 11',
    'Unnamed: 12'
]

df = df.drop(
    columns=[c for c in unused_cols if c in df.columns],
    errors='ignore'
)

df = df.fillna('')


# =========================================================
# HIERARCHY
# =========================================================

def create_hierarchy(row):

    hierarchy = []

    if row["heading_1"].strip():
        hierarchy.append(row["heading_1"].strip())

    if row["heading_2"].strip():
        hierarchy.append(row["heading_2"].strip())

    if row["heading_3"].strip():
        hierarchy.append(row["heading_3"].strip())

    return " > ".join(hierarchy)


df["hierarchy_key"] = df.apply(
    create_hierarchy,
    axis=1
)

print('Hierarchy Created')

# =========================================================
# CREATE PARENT DOCS
# =========================================================

parent_documents = []

grouped = df.groupby("hierarchy_key")

for hierarchy_key, group_df in grouped:

    combined_text = ""

    for _, row in group_df.iterrows():

        combined_text += f"""

Subsection:
{row.get('heading_4', '')}

Nested Section:
{row.get('heading_5', '')}

Section Number:
{row.get('sub_section_no', '')}

Content:
{row.get('text', '')}

"""

    parent_content = f"""

Hierarchy:
{hierarchy_key}

{combined_text}

"""

    parent_id = str(uuid.uuid4())

    doc = Document(

        page_content=parent_content,

        metadata={
            "parent_id": parent_id,
            "hierarchy_key": hierarchy_key
        }
    )

    parent_documents.append(doc)


# =========================================================
# SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=700,
    chunk_overlap=100,

    separators=[
        "\nSection Number:",
        "\nSubsection:",
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)


# =========================================================
# FINAL CHUNKS
# =========================================================

final_docs = []

for doc in parent_documents:

    if len(doc.page_content) < 1200:

        final_docs.append(doc)

    else:

        child_chunks = text_splitter.split_documents([doc])

        final_docs.extend(child_chunks)


# =========================================================
# EMBEDDINGS
# =========================================================

embedding_model = HuggingFaceEmbeddings(

    model_name="BAAI/bge-small-en-v1.5",

    model_kwargs={
        'device': 'cpu'
    },

    encode_kwargs={
        'normalize_embeddings': True
    }
)

print('embedding_model Loaded')

# =========================================================
# VECTOR DB
# =========================================================

# vectordb = Chroma.from_documents(

#     documents=final_docs,

#     embedding=embedding_model,

#     persist_directory="vectordb/chroma_db"
# )


vectordb = Chroma(

    persist_directory="vectordb/chroma_db",

    embedding_function=embedding_model
)

print('Vector DB created')

# =========================================================
# LLM
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(

    model_name="llama-3.3-70b-versatile",

    temperature=0,

    groq_api_key=GROQ_API_KEY
)

print('LLM model loaded')
# =========================================================
# ASK QUESTION
# =========================================================
def ask_question(question):

    formatted_query = (
        "Represent this legal question for retrieving relevant passages: "
        + question
    )

    # =========================================================
    # TOP-K RETRIEVAL
    # =========================================================

    results = vectordb.similarity_search_with_score(
        formatted_query,
        k=8
    )

    # keep docs + scores (IMPORTANT FIX)
    docs_with_scores = results

    # =========================================================
    # BUILD STRUCTURED CONTEXT
    # =========================================================

    context = ""

    docs = []

    for i, (doc, score) in enumerate(docs_with_scores, 1):

        docs.append(doc)

        context += f"""
[Document {i}]
Relevance Score: {score}

Hierarchy: {doc.metadata.get("hierarchy_key", "")}

Content:
{doc.page_content}

--------------------------------------

"""

    # =========================================================
    # CITATIONS (UNCHANGED BUT SAFE)
    # =========================================================

    citations = []
    seen = set()

    for doc in docs:

        hierarchy = doc.metadata.get("hierarchy_key", "")

        if hierarchy in seen:
            continue

        seen.add(hierarchy)

        parts = [p.strip() for p in hierarchy.split(">")]

        citations.append({
            "Parent_Section": parts[0] if len(parts) > 0 else "",
            "Chapter": parts[1] if len(parts) > 1 else "",
            "Sub_Section": parts[2] if len(parts) > 2 else ""
        })

    # =========================================================
    # STRONGER PROMPT (FORCES MULTI-DOC USAGE)
    # =========================================================

    prompt = f"""
You are an AI legal assistant.

You MUST use ALL provided documents to answer the question.
Do not rely on only one passage.

If multiple documents contain relevant information, combine them into a single structured answer.

RULES:
- Do NOT hallucinate
- Do NOT use outside knowledge
- If answer is not found, say:
  "I could not find that information in the documents."

CONTEXT DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    # =========================================================
    # LLM CALL
    # =========================================================

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "citations": citations
    }