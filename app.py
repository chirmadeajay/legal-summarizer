import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Load embedding model ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Extract text from PDF ---
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Split text into chunks ---
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- Build FAISS index ---
def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# --- Retrieve top chunks ---
def retrieve(query, index, chunks, embedder, top_k=5):
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# --- Summarize with Groq ---
def summarize(context, query):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal document analyst.
Given the following excerpts from a legal judgement, answer the query clearly and concisely.
Structure your response with these sections:
- **Summary**
- **Key Points**
- **Final Verdict / Outcome**

Context:
{context}"""),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return response.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Legal Summarizer", page_icon="⚖️")
st.title("⚖️ Legal Judgement Summarizer")
st.caption("Upload a legal PDF and ask questions or get a summary")

embedder = load_embedder()

uploaded_file = st.file_uploader("Upload Legal PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = build_index(chunks, embedder)

    st.success(f"Document loaded — {len(chunks)} chunks indexed")

    query = st.text_input(
        "Ask a question or request a summary:",
        placeholder="e.g. What is the final verdict? Summarize this judgement."
    )

    if st.button("Analyse") and query:
        with st.spinner("Retrieving relevant sections..."):
            top_chunks = retrieve(query, index, chunks, embedder)
            context = "\n\n".join(top_chunks)

        with st.spinner("Generating summary..."):
            result = summarize(context, query)

        st.subheader("📄 Analysis")
        st.markdown(result)

        with st.expander("🔍 Retrieved Context"):
            st.text(context)