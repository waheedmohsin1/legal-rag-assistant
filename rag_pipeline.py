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

    # if row["heading_3"].strip():
    #     hierarchy.append(row["heading_3"].strip())

    return " > ".join(hierarchy)


df["hierarchy_key"] = df.apply(
    create_hierarchy,
    axis=1
)

print("\nHierarchy Created")



print('Hierarchy Created')

# =========================================================
# CREATE PARENT DOCS
# =========================================================

parent_documents = []
child_documents = []

grouped = df.groupby("hierarchy_key")

# CHAPTER SPLITTER (IMPORTANT)
chapter_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". "
    ]
)

for hierarchy_key, group_df in grouped:

    chapter_grouped = group_df.groupby("heading_2")

    for chapter_name, chapter_df in chapter_grouped:

        # STEP 1: build full chapter text
        chapter_text = ""

        for _, row in chapter_df.iterrows():
            chapter_text += row.get("text", "") + "\n\n"

        # STEP 2: split chapter into smaller chunks
        chunks = chapter_splitter.split_text(chapter_text)

        parent_id = str(uuid.uuid4())

        # STEP 3: create ONE parent record (chapter-level metadata)
        parent_documents.append(
            Document(
                page_content=f"""
                    Hierarchy:
                    {hierarchy_key}

                    Chapter:
                    {chapter_name}
                    """,
                metadata={
                    "parent_id": parent_id,
                    "hierarchy_key": hierarchy_key,
                    "chapter": chapter_name
                }
            )
        )

        # STEP 4: create child chunks (FOR VECTOR DB)
        for i, chunk in enumerate(chunks):

            child_documents.append(
                Document(
                    page_content=f"""
                        Hierarchy:
                        {hierarchy_key}

                        Chapter:
                        {chapter_name}

                        Content:
                        {chunk}
""",
                    metadata={
                        "parent_id": parent_id,
                        "hierarchy_key": hierarchy_key,
                        "chapter": chapter_name,
                        "chunk_id": i
                    }
                )
            )

print("Parents:", len(parent_documents))
print("Children:", len(child_documents))

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

vectordb = Chroma.from_documents(

    documents=child_documents,

    embedding=embedding_model,

    persist_directory="vectordb/chroma_db"
)
                                       # ********************************************
# If embeddings are created once then comment above code and uncomment below code
                                       # **********************************************
# vectordb = Chroma(

#     persist_directory="vectordb/chroma_db",

#     embedding_function=embedding_model
# )

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

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    formatted_query = (
        "Represent this legal question for retrieving relevant passages: "
        + question
    )

    # =====================================================
    # STEP 1: VECTOR SEARCH
    # =====================================================
    results = vectordb.similarity_search_with_score(
        formatted_query,
        k=8
    )

    print("\nRETRIEVED DOCUMENTS\n")

    # =====================================================
    # STEP 2: GROUP BY parent_id
    # =====================================================
    clusters = {}

    for doc, score in results:

        parent_id = doc.metadata.get("parent_id")

        if parent_id not in clusters:
            clusters[parent_id] = {
                "docs": [],
                "score": 0
            }

        clusters[parent_id]["docs"].append((doc, score))

        # accumulate score (lower is better for Chroma usually)
        clusters[parent_id]["score"] += score

        print("SCORE:", score)
        print("HIERARCHY:", doc.metadata.get("hierarchy_key"))
        print("-" * 50)

    # =====================================================
    # STEP 3: SELECT BEST CLUSTER
    # =====================================================

    best_parent_id = min(
        clusters.keys(),
        key=lambda pid: clusters[pid]["score"]
    )

    selected_docs_with_scores = clusters[best_parent_id]["docs"]

    # =====================================================
    # STEP 4: LOCAL EXPANSION (OPTIONAL)
    # =====================================================

    selected_docs_with_scores.sort(key=lambda x: x[1])

    final_docs = [doc for doc, _ in selected_docs_with_scores]

    # OPTIONAL: limit context size (VERY IMPORTANT)
    context_parts = []
    total_length = 0
    MAX_CONTEXT = 6000

    for doc in final_docs:

        text = doc.page_content

        if total_length + len(text) > MAX_CONTEXT:
            break

        context_parts.append(text)
        total_length += len(text)

    context = "\n\n".join(context_parts)

    print("\n" + "=" * 80)
    print("CONTEXT SAMPLE")
    print("=" * 80)
    print(context[:3000])

    # =====================================================
    # STEP 5: CITATIONS
    # =====================================================

    citations = []
    seen = set()

    for doc, _ in selected_docs_with_scores:

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

    # =====================================================
    # STEP 6: PROMPT
    # =====================================================

    prompt = f"""
You are an AI legal assistant.

Answer ONLY from provided context.

RULES:
- Do NOT hallucinate
- Do NOT invent information
- If answer unavailable say:
  "I could not find that information in the documents."
- Keep answer concise
- Mention legal sections when relevant

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "citations": citations,
        "retrieved_docs": final_docs
    }