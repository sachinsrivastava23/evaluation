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

# Load secrets from Streamlit Cloud
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
        self.chunk = all_chunks
        self.total_count = len(all_chunks)
        self.used_indices = set()

    def get_next_unique_chunk(self):
        all_indices = set(range(self.total_count))
        available = list(all_indices - self.used_indices)

        if not available:
            return "all chunks have been used"

        idx = random.choice(available)
        self.used_indices.add(idx)
        return self.chunk[idx]

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
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Google Search API Error: {e}")
        return {}

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )

    final_chunks = []
    for d in docs:
        text = re.sub(r"\s+", " ", d["page_content"])
        text = text.encode("ascii", "ignore").decode()

        chunks = text_splitter.split_text(text)
        for c in chunks:
            if c.strip():
                final_chunks.append({"page_content": c})

    return final_chunks

def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#          PDF TEXT AUTO-DETECTOR (NO OCR)
# ------------------------------------------------------

def load_pdf_auto(path: str):
    reader = PdfReader(path)

    for page in reader.pages:
        if page.extract_text():
            docs = PyPDFLoader(path).load()
            return [{"page_content": d.page_content} for d in docs]

    return [{"page_content": "PDF has no readable text."}]

# ------------------------------------------------------
#        RAW CHROMA CLIENT (NO INTERNAL CRASHES)
# ------------------------------------------------------

import chromadb
from chromadb.config import Settings

CHROMA_PATH = "chroma_db"

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
    )
)

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

st.title("Upload the PDF file")

uploaded_file = st.file_uploader("Upload your material", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    st.success("PDF Uploaded!")
    st.write("Processing… ⏳")

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
    "Select how you want questions:",
    ("Generated From the Notes", "PYQ")
)

# ------------------------------------------------------
#     MODE 1: GENERATED FROM NOTES
# ------------------------------------------------------

if mode == "Generated From the Notes":

    if "generated_question" not in st.session_state:
        st.session_state.generated_question = None

    if "current_chunk" not in st.session_state:
        st.session_state.current_chunk = None

    if "user_ans" not in st.session_state:
        st.session_state.user_ans = ""

    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    if st.button("Generate Question"):
        selector = UniqueChunkSelector(chunk)
        chunk_text = selector.get_next_unique_chunk()
        st.session_state.current_chunk = chunk_text

        prompt = f"""
        Create 1 exam-style question STRICTLY from the text below.

        TEXT:
        {chunk_text}
        """

        response = model.invoke(prompt)
        st.session_state.generated_question = response.content

        st.session_state.user_ans = ""
        st.session_state.evaluation = None

    if st.session_state.generated_question:
        st.write("### Question:")
        st.write(st.session_state.generated_question)

        st.session_state.user_ans = st.text_area(
            "Write your answer:",
            value=st.session_state.user_ans
        )

        if st.button("Submit Answer"):
            evaluation_prompt = f"""
            Evaluate the answer based ONLY on the material.

            Question: {st.session_state.generated_question}
            Answer: {st.session_state.user_ans}
            Material: {st.session_state.current_chunk}

            Provide:
            - Score (0-10)
            - What is correct
            - What is missing
            - Improved answer
            """

            ev = model.invoke(evaluation_prompt)
            st.session_state.evaluation = ev.content

    if st.session_state.evaluation:
        st.write("### Evaluation:")
        st.write(st.session_state.evaluation)

# ------------------------------------------------------
#           MODE 2: PYQ EXTRACTION + EVALUATION
# ------------------------------------------------------

if mode == "PYQ":

    if "pyq_list" not in st.session_state:
        st.session_state.pyq_list = []

    if "pyq_index" not in st.session_state:
        st.session_state.pyq_index = 0

    st.info("📘 Searching ABESIT Question Bank")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox("Course", ["B.Tech", "MCA", "MBA", "BBA", "BCA"])
        year = st.text_input("Year", placeholder="2022")

    with col2:
        subject = st.text_input("Subject Code/Name", placeholder="KCS301")
        st.caption("TIP: Use subject codes like KCS-301")

    if st.button("Search ABESIT"):

        if not subject:
            st.error("Enter subject!")
        else:
            q = f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf'
            data = google_search(q)

            urls = [x["link"] for x in data.get("items", []) if "link" in x]

            if not urls:
                st.error("No PDFs found!")
                st.stop()

            all_text = ""
            headers = {"User-Agent": "Mozilla/5.0"}

            for link in urls:
                if link.endswith(".pdf"):
                    try:
                        r = requests.get(link, headers=headers, timeout=8)

                        with open("temp.pdf", "wb") as f:
                            f.write(r.content)

                        reader = PdfReader("temp.pdf")
                        t = ""

                        for p in reader.pages:
                            text = p.extract_text()
                            if text:
                                t += text + "\n"

                        if t.strip():
                            all_text += t
                    except:
                        pass

            extract_prompt = f"""
            Extract ONLY exam questions from text below.

            TEXT:
            {all_text[:30000]}

            Return Python list: ["Q1", "Q2"]
            """

            resp = model.invoke(extract_prompt)
            clean = re.sub(r"```[a-zA-Z]*", "", resp.content).replace("```", "")

            try:
                pyqs = ast.literal_eval(clean)
            except:
                try: pyqs = json.loads(clean)
                except:
                    st.error("Parsing failed.")
                    st.stop()

            st.session_state.pyq_list = pyqs
            st.session_state.pyq_index = 0
            st.success(f"Found {len(pyqs)} questions!")
            st.rerun()

    if st.session_state.pyq_list:

        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        st.write(f"### Question {idx+1}/{total}")
        st.write(st.session_state.pyq_list[idx])

        ans = st.text_area("Your Answer:")

        if st.button("Evaluate"):
            retrieved = chroma_search(st.session_state.pyq_list[idx], k=3)
            text = "\n\n".join(retrieved)

            eval_prompt = f"""
            Evaluate the answer.

            Question: {st.session_state.pyq_list[idx]}
            Answer: {ans}
            Notes: {text}

            Provide:
            - Score /10
            - Correct parts
            - Missing parts
            - Improved answer
            """

            res = model.invoke(eval_prompt)
            st.write(res.content)

        if st.button("Next"):
            st.session_state.pyq_index += 1
            st.rerun()
