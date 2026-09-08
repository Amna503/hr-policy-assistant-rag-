import io
import os
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #111827,
            #172033
        );
        border: 1px solid #263244;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #aab4c3;
        font-size: 1rem;
    }

    .source-card {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background-color: #151b26;
        border: 1px solid #293447;
        margin-top: 0.5rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        background-color: #151b26;
        border: 1px solid #293447;
        text-align: center;
    }

    .small-text {
        color: #8d99aa;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = None

    # Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Fallback for local development
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file) -> List[Dict]:

    file_bytes = uploaded_file.getvalue()

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf):

        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "source": uploaded_file.name,
                    "page": page_number + 1,
                }
            )

    pdf.close()

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": page_data["source"],
                        "page": page_data["page"],
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def build_faiss_index(chunks: List[Dict]):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# PROCESS UPLOADED PDFs
# ============================================================

def process_pdfs(uploaded_files):

    all_chunks = []

    for uploaded_file in uploaded_files:

        pages = extract_pdf_text(
            uploaded_file
        )

        chunks = create_chunks(
            pages
        )

        all_chunks.extend(chunks)

    if not all_chunks:
        return None, []

    index = build_faiss_index(
        all_chunks
    )

    return index, all_chunks


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_documents(
    question: str,
    index,
    chunks: List[Dict],
    top_k: int = TOP_K,
):

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(
    question: str,
    retrieved_documents: List[Dict],
):

    client = get_groq_client()

    if client is None:

        return (
            "⚠️ Groq API key is not configured. "
            "Please add `GROQ_API_KEY` to Streamlit Cloud Secrets."
        )

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {document['source']}
Page: {document['page']}

Content:
{document['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Your task is to answer questions using ONLY the
HR policy content provided in the context.

STRICT RULES:

1. Do not invent or assume HR policies.
2. Do not use outside knowledge.
3. If the answer is not supported by the provided
   policy context, clearly say:

   "I couldn't find this information in the
   uploaded HR policies."

4. Give concise, professional answers.
5. Mention relevant policy documents and page numbers
   when answering.
6. If multiple policies are relevant, explain the
   relevant information clearly.
7. If a policy appears ambiguous or incomplete,
   recommend contacting HR.
8. Do not provide legal advice.
9. Do not claim to be a human HR employee.
10. Never fabricate a source or page number.

Answer only from the supplied context.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 HR Knowledge Base")

    st.markdown(
        """
        Upload your company's HR policy PDFs.
        
        The assistant will extract the text,
        create embeddings, and index the
        policies using FAISS.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload HR Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more HR policy PDF files.",
    )

    st.divider()

    if uploaded_files:

        st.write(
            f"📄 **{len(uploaded_files)} PDF(s) selected**"
        )

        for file in uploaded_files:

            st.caption(
                f"• {file.name}"
            )

        if st.button(
            "🔍 Process Policies",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner(
                "Reading and indexing HR policies..."
            ):

                try:

                    index, chunks = process_pdfs(
                        uploaded_files
                    )

                    if index is None:

                        st.error(
                            "No readable text was found "
                            "in the uploaded PDFs."
                        )

                    else:

                        st.session_state.index = index

                        st.session_state.chunks = chunks

                        st.session_state.documents_processed = True

                        st.session_state.uploaded_files = [
                            file.name
                            for file in uploaded_files
                        ]

                        st.session_state.chat_history = []

                        st.success(
                            f"Processed {len(chunks)} policy chunks."
                        )

                except Exception as error:

                    st.error(
                        f"Processing failed: {error}"
                    )

    if st.session_state.documents_processed:

        st.divider()

        st.success(
            "✅ Knowledge base ready"
        )

        st.metric(
            "Policy Chunks",
            len(st.session_state.chunks)
        )

        if st.button(
            "🗑️ Clear Knowledge Base",
            use_container_width=True,
        ):

            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.documents_processed = False
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []

            st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <h1>🧑‍💼 HR Policy Assistant</h1>

    <p>
    Upload your HR policy PDFs and ask questions.
    The assistant retrieves relevant policy sections
    using semantic search and generates answers with
    source references.
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STATUS
# ============================================================

if not st.session_state.documents_processed:

    st.info(
        "👈 Upload your HR policy PDFs from the sidebar "
        "and click **Process Policies** to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
            <h3>📄 Upload</h3>
            <p class="small-text">
            Upload multiple HR policy PDFs.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
            <h3>🔎 Retrieve</h3>
            <p class="small-text">
            FAISS finds relevant policy sections.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
            <h3>🤖 Answer</h3>
            <p class="small-text">
            Groq generates grounded answers.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.success(
        f"Knowledge base contains "
        f"**{len(st.session_state.chunks)} chunks** "
        f"from "
        f"**{len(st.session_state.uploaded_files)} PDF(s)**."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📚 View sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        **{source['source']}** — "
                        Page {source['page']}  
                        Relevance: {source['score']:.2f}
                        """
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your HR policies..."
)


if question:

    if not st.session_state.documents_processed:

        st.warning(
            "Please upload and process your HR policy PDFs first."
        )

        st.stop()

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your HR policies..."
        ):

            try:

                retrieved_documents = retrieve_documents(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                )

                answer = generate_answer(
                    question,
                    retrieved_documents,
                )

                st.markdown(answer)

                sources = []

                for document in retrieved_documents:

                    source_key = (
                        document["source"],
                        document["page"],
                    )

                    if source_key not in [
                        (
                            source["source"],
                            source["page"],
                        )
                        for source in sources
                    ]:

                        sources.append(
                            {
                                "source": document["source"],
                                "page": document["page"],
                                "score": document["score"],
                            }
                        )

                with st.expander(
                    "📚 View retrieved sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"""
                            **{source['source']}**

                            Page: {source['page']}

                            Relevance score:
                            `{source['score']:.3f}`
                            """
                        )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(error_message)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
