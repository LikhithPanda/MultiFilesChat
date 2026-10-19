"""
MultiPDFChat - Streamlit Web App
Chat with your files using Google Gemini + LangChain + FAISS.

Run locally:
    streamlit run app.py
"""

import os
import asyncio

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

from file_loaders import load_file, SUPPORTED_EXTENSIONS


# ----------------------------------------------------------------------
# Streamlit runs each script in a worker thread ("ScriptRunner.scriptThread")
# that has no asyncio event loop by default. The Google GenAI client needs one, 
# so this creates and registers one before any Gemini call happens.
# Without this, embedding/chat calls fail with:
#   RuntimeError: There is no current event loop in thread 'ScriptRunner.scriptThread'
# ----------------------------------------------------------------------
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ----------------------------------------------------------------------
# Embedding model name.
# Newer langchain-google-genai versions require the full "models/..." path
# rather than the bare model name - a bare name like "gemini-embedding-001"
# fails with "unexpected model name format".
# ----------------------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
# If your API key doesn't have access to that model, switch to the older,
# broadly-available one instead:
# EMBEDDING_MODEL = "models/embedding-001"


# PAGE CONFIG
st.set_page_config(page_title="MultiPDFChat", page_icon="📄", layout="wide")


# API KEY HANDLING
def get_api_key():
    """
    Looks for the Gemini API key in this order:
    1. st.secrets["GOOGLE_API_KEY"]   (used on Streamlit Community Cloud)
    2. the GOOGLE_API_KEY environment variable (used for local/Docker runs)
    3. a text box in the sidebar, as a last resort so the app never hardcodes a key
    """
    key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = st.session_state.get("manual_api_key")
    return key

def ensure_api_key_configured():
    key = get_api_key()
    if not key:
        st.sidebar.warning("No Gemini API key found in secrets or environment.")
        manual_key = st.sidebar.text_input(
            "Enter your Gemini API key", type="password", key="manual_api_key"
        )
        if manual_key:
            key = manual_key
            st.rerun()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        genai.configure(api_key=key)
    return key


# FILES -> TEXT -> CHUNKS -> FAISS
def build_chunks(uploaded_files):
    """
    Reads every uploaded file (PDF, CSV, XLSX, DOCX, PPTX, code, text, SQLite...)
    via file_loaders.load_file, splits each into chunks, and tags every chunk
    with its source filename so answers can say where they came from.
    Returns (texts, metadatas, notes, failed_files).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts, metadatas, notes, failed = [], [], [], []
    for f in uploaded_files:
        try:
            text, file_notes = load_file(f.name, f.getvalue())
        except Exception as e:  # one bad file shouldn't sink the whole batch
            failed.append(f"{f.name}: {e}")
            continue
        notes.extend(file_notes)
        if not text.strip():
            failed.append(f"{f.name}: no readable text found")
            continue
        for chunk in splitter.split_text(text):
            # Prefix the source so the LLM also sees which file a chunk is from.
            texts.append(f"[Source: {f.name}]\n{chunk}")
            metadatas.append({"source": f.name})
    return texts, metadatas, notes, failed

def build_vector_store(texts, metadatas):
    """
    Builds an in-memory FAISS index and returns it directly, instead of
    saving to / loading from disk, so the app works with whatever files the
    current user uploaded without needing persistent storage.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

def get_conversational_chain():
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible using ONLY
    the information provided in the context.

    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Do not use outside knowledge
    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return prompt | model | StrOutputParser()

def answer_question(vector_store, user_question):
    docs = vector_store.similarity_search(user_question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = get_conversational_chain()
    answer = chain.invoke({"context": context, "question": user_question})
    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    return answer, sources


# STREAMLIT UI
def main():
    st.title("Chat with Your Files (Gemini + RAG)")
    st.caption("Upload PDFs, spreadsheets, documents, slides, code or databases, then ask questions grounded only in their content.")
    api_key = ensure_api_key_configured()
    # Session state holds the FAISS index and chat history so they survive
    # reruns (Streamlit reruns the whole script on every interaction).
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (question, answer) tuples
    if "indexed_file_names" not in st.session_state:
        st.session_state.indexed_file_names = []
    with st.sidebar:
        st.header("1. Upload files")
        uploaded_files = st.file_uploader(
            "Choose one or more files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="PDF, CSV, XLSX, DOCX, PPTX, TXT, Markdown, SQLite (.db/.sqlite), "
                 "and source code such as .py .java .php .sql .js and more.",
        )
        process_clicked = st.button("Process files", type="primary", disabled=not api_key)
        if not api_key:
            st.info("Enter your Gemini API key above to enable processing.")
        if process_clicked:
            if not uploaded_files:
                st.error("Please upload at least one file first.")
            else:
                with st.spinner("Reading files, splitting into chunks, and building the index..."):
                    texts, metadatas, notes, failed = build_chunks(uploaded_files)
                    for message in notes:
                        st.warning(message)
                    for message in failed:
                        st.error(f"Skipped {message}")
                    if not texts:
                        st.error("No usable text was found in the uploaded files. "
                                 "(Scanned PDFs without a text layer can't be read.)")
                    else:
                        st.session_state.vector_store = build_vector_store(texts, metadatas)
                        st.session_state.indexed_file_names = sorted({m["source"] for m in metadatas})
                        st.session_state.chat_history = []
                        st.success(
                            f"Indexed {len(st.session_state.indexed_file_names)} file(s) "
                            f"into {len(texts)} chunks."
                        )
        if st.session_state.indexed_file_names:
            st.markdown("**Currently indexed:**")
            for name in st.session_state.indexed_file_names:
                st.markdown(f"- {name}")
    st.header("2. Ask a question")
    if st.session_state.vector_store is None:
        st.info("Upload file(s) and click **Process files** in the sidebar to get started.")
        return
    user_question = st.chat_input("Ask a question about your files...")
    if user_question:
        with st.spinner("Searching your documents and asking Gemini..."):
            answer, sources = answer_question(st.session_state.vector_store, user_question)
        st.session_state.chat_history.append((user_question, answer, sources))
    # Render chat history, most recent last (standard chat order)
    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                st.caption("Sources searched: " + ", ".join(sources))


if __name__ == "__main__":
    main()