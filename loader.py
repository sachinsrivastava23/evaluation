import os
import random
import tempfile
import json
import requests
import re
import ast

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# ------------------------------------------------------
#                  INITIAL SETUP
# ------------------------------------------------------

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Load secrets (Streamlit Cloud)
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ------------------------------------------------------
#                EMBEDDING MODEL
# ------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# ------------------------------------------------------
#          UNIQUE CHUNK SELECTOR CLASS
# ------------------------------------------------------

class UniqueChunkSelector:
    def __init__(self, all_chunks):
        self.chunks = all_chunks
        self.used = set()

    def get(self):
        available = list(set(range(len(self.chunks))) - self.used)
        if not available:
            return "All chunks used."

        idx = random.choice(available)
        self.used.add(idx)
        return self.chunks[idx]

# ------------------------------------------------------
#            GOOGLE SEARCH API FUNCTION
# ------------------------------------------------------

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": num_results
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Google Search API Error: {e}")
        return {}

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    final = []

    for d in docs:
        text = d["page_content"]
        text = re.sub(r"\s+", " ", text)
        text = text.encode("ascii", "ignore").decode()

        for c in splitter.split_text(text):
            if c.strip():
                final.append({"page_content": c})
    return final

def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#          PDF TEXT AUTO-DETECTOR (NO OCR)
# ------------------------------------------------------

def load_pdf_auto(path):
    reader = PdfReader(path)
    for page in reader.pages:
        if page.extract_text():
            docs = PyPDFLoader(path).load()
            return [{"page_content": d.page_content} for d in docs]
    return [{"page_content": "PDF has no readable text."}]

# ------------------------------------------------------
#        RAW CHROMA CLIENT (LATEST API — STABLE)
# ------------------------------------------------------

from chromadb import PersistentClient

CHROMA_PATH = "chroma_db"

client = PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=embeddings,
)

def add_to_chroma(docs):
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc.page_content],
            ids=[f"doc_{i}"]
        )
    client.persist()

def chroma_search(query, k=3):
    res = collection.query(
        query_texts=[query],
        n_results=k
    )
    return res["documents"][0] if res["documents"] else []

# ------------------------------------------------------
#                    STREAMLIT UI
# ------------------------------------------------------

st.title("📘 Smart Study Companion")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    st.success("PDF Uploaded")
    st.info("Processing… ⏳")

    docs = load_pdf_auto(path)
    chunk = chunk_documents(docs)
    chunk_docs = to_document(chunk)

    add_to_chroma(chunk_docs)

    st.success("Embeddings stored successfully!")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.subheader("Choose Question Type")

mode = st.radio(
    "Select mode:",
    ("Generated From the Notes", "PYQ")
)

# ------------------------------------------------------
#     MODE 1: GENERATED FROM NOTES
# ------------------------------------------------------

if mode == "Generated From the Notes":

    if "generated_question" not in st.session_state:
        st.session_state.generated_question = None

    if "chunk_text" not in st.session_state:
        st.session_state.chunk_text = None

    if st.button("Generate Question"):
        selector = UniqueChunkSelector(chunk)
        chunk_text = selector.get()
        st.session_state.chunk_text = chunk_text

        prompt = f"""
        Generate ONE exam question strictly from the material below:

        TEXT:
        {chunk_text}
        """

        resp = model.invoke(prompt)
        st.session_state.generated_question = resp.content

    if st.session_state.generated_question:
        st.write("### Question:")
        st.write(st.session_state.generated_question)

        answer = st.text_area("Write your answer:")

        if st.button("Evaluate Answer"):
            eval_prompt = f"""
            Evaluate the answer strictly based on the study material.

            Question: {st.session_state.generated_question}
            Answer: {answer}
            Material: {st.session_state.chunk_text}

            Provide:
            - Score (0-10)
            - Correct parts
            - Missing parts
            - Improved answer
            """

            res = model.invoke(eval_prompt)
            st.write(res.content)

# ------------------------------------------------------
#           MODE 2: PYQ EXTRACTION + EVALUATION
# ------------------------------------------------------

if mode == "PYQ":

    if "pyq_list" not in st.session_state:
        st.session_state.pyq_list = []

    if "pyq_index" not in st.session_state:
        st.session_state.pyq_index = 0

    st.info("🔎 Search ABESIT Question Bank")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox("Course", ["B.Tech", "MCA", "MBA", "BBA", "BCA"])
        year = st.text_input("Year (optional)", placeholder="2022")

    with col2:
        subject = st.text_input("Subject Code/Name", placeholder="KCS301")
        st.caption("Use subject codes for better results.")

    if st.button("Search PYQs"):

        if not subject:
            st.error("Enter a subject.")
        else:
            q = f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf'
            data = google_search(q)

            urls = [i["link"] for i in data.get("items", []) if "link" in i]

            if not urls:
                st.error("No PDFs found.")
                st.stop()

            text = ""
            for link in urls:
                if link.endswith(".pdf"):
                    try:
                        r = requests.get(link, timeout=8)
                        with open("temp.pdf", "wb") as f:
                            f.write(r.content)

                        reader = PdfReader("temp.pdf")
                        for p in reader.pages:
                            t = p.extract_text()
                            if t:
                                text += t + "\n"
                    except:
                        pass

            extract_prompt = f"""
            Extract ONLY exam questions from the text:

            TEXT:
            {text[:30000]}

            Return Python list: ["Q1", "Q2"]
            """

            r = model.invoke(extract_prompt)
            clean = r.content.replace("```", "")
            clean = re.sub(r"```[a-zA-Z]*", "", clean)

            try:
                pyqs = ast.literal_eval(clean)
            except:
                try:
                    pyqs = json.loads(clean)
                except:
                    st.error("Failed to parse questions.")
                    st.stop()

            st.session_state.pyq_list = pyqs
            st.session_state.pyq_index = 0
            st.success(f"Found {len(pyqs)} questions!")
            st.rerun()

    if st.session_state.pyq_list:
        idx = st.session_state.pyq_index
        q = st.session_state.pyq_list[idx]

        st.write(f"### Question {idx+1}:")
        st.write(q)

        answer = st.text_area("Your Answer:")

        if st.button("Evaluate"):
            retrieved = chroma_search(q, k=3)
            context = "\n\n".join(retrieved)

            eval_prompt = f"""
            Evaluate the answer using the notes.

            Question: {q}
            Answer: {answer}
            Notes: {context}

            Give:
            - Score /10
            - Correct points
            - Missing points
            - Best possible answer
            """

            res = model.invoke(eval_prompt)
            st.write(res.content)

        if st.button("Next"):
            st.session_state.pyq_index += 1
            st.rerun()
