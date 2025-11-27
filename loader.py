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
from langchain_community.vectorstores import Qdrant

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ------------------------------------------------------
#                  INITIAL SETUP
# ------------------------------------------------------

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Load secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Connect to Qdrant Cloud
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Create collection if not exists
COLLECTION_NAME = "study_notes"

try:
    qdrant_client.get_collection(COLLECTION_NAME)
except:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=768,
            distance=qmodels.Distance.COSINE
        )
    )

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
        return r.json()
    except:
        return {}

# ------------------------------------------------------
#                EMBEDDING MODEL
# ------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final = []
    for d in docs:
        parts = splitter.split_text(d["page_content"])
        for c in parts:
            final.append({"page_content": c})
    return final

def to_documents(chunk_list):
    return [Document(page_content=c["page_content"]) for c in chunk_list]

# ------------------------------------------------------
#             LOAD PDF WITHOUT OCR
# ------------------------------------------------------

def load_pdf_auto(path):
    reader = PdfReader(path)
    for page in reader.pages:
        txt = page.extract_text()
        if txt and txt.strip():
            docs = [{"page_content": p.extract_text()} for p in reader.pages]
            return docs
    return [{"page_content": "NO TEXT FOUND"}]

# ------------------------------------------------------
#                    STREAMLIT UI
# ------------------------------------------------------

st.title("Upload PDF")

uploaded_file = st.file_uploader("Upload your material", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
        t.write(uploaded_file.read())
        pdf_path = t.name

    st.success("Uploaded! Processing…")

    docs = load_pdf_auto(pdf_path)
    chunks = chunk_documents(docs)
    chunk_docs = to_documents(chunks)

    # Store chunks in Qdrant Cloud
    Qdrant.from_documents(
        documents=chunk_docs,
        embedding=embeddings,
        client=qdrant_client,
        collection_name=COLLECTION_NAME
    )

    st.success("Embeddings stored successfully!")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.subheader("Choose Question Type")
mode = st.radio("", ("Generated From Notes", "PYQ"))

# ------------------------------------------------------
#      MODE 1 — Question From Notes
# ------------------------------------------------------

if mode == "Generated From Notes":

    if st.button("Generate Question"):
        random_chunk = random.choice(chunks)

        prompt = f"""
        Generate one exam-style question ONLY from the material below:

        {random_chunk["page_content"]}
        """

        out = model.invoke(prompt)
        st.session_state.generated_question = out.content
        st.session_state.answer = ""
        st.session_state.eval = None
        st.session_state.current_chunk = random_chunk["page_content"]

    if "generated_question" in st.session_state:
        st.write("### Question:")
        st.write(st.session_state.generated_question)

        st.session_state.answer = st.text_area("Write your answer:")

        if st.button("Evaluate"):
            eval_prompt = f"""
            Question: {st.session_state.generated_question}
            Answer: {st.session_state.answer}
            Material: {st.session_state.current_chunk}

            Give:
            - Score (0-10)
            - What is correct
            - What is missing
            - Correct answer
            """

            res = model.invoke(eval_prompt)
            st.session_state.eval = res.content

    if st.session_state.get("eval"):
        st.write("### Evaluation:")
        st.write(st.session_state.eval)

# ------------------------------------------------------
#      MODE 2 — PYQ Extraction + Evaluation
# ------------------------------------------------------

if mode == "PYQ":
    st.info("Scanning ABESIT…")

    subject = st.text_input("Subject")
    year = st.text_input("Year")

    if st.button("Search PYQ"):
        query = f'site:abesit.in "{subject}" "{year}" filetype:pdf'
        data = google_search(query)

        links = [i["link"] for i in data.get("items", [])]
        if not links:
            st.warning("No PDFs found!")
            st.stop()

        all_text = ""
        for link in links:
            if link.endswith(".pdf"):
                try:
                    r = requests.get(link)
                    with open("temp.pdf", "wb") as f:
                        f.write(r.content)

                    reader = PdfReader("temp.pdf")
                    for p in reader.pages:
                        txt = p.extract_text()
                        if txt:
                            all_text += txt + "\n"
                except:
                    pass

        extract_prompt = f"""
        Extract QUESTIONS ONLY from this text and return a Python list:

        {all_text[:25000]}
        """

        res = model.invoke(extract_prompt)
        cleaned = res.content.strip()

        try:
            st.session_state.pyq = ast.literal_eval(cleaned)
        except:
            st.error("Error extracting questions.")
            st.stop()

        st.success(f"Found {len(st.session_state.pyq)} questions!")

    if "pyq" in st.session_state:
        idx = st.session_state.get("pyq_idx", 0)
        q = st.session_state.pyq[idx]

        st.write(f"### Question {idx+1}")
        st.write(q)

        ans = st.text_area("Your Answer")

        if st.button("Evaluate Answer"):

            # RAG retrieval using Qdrant Cloud
            rag_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embeddings.embed_query(q),
                limit=3
            )

            context = "\n\n".join(hit.payload.get("page_content", "") for hit in rag_results)

            eval_prompt = f"""
            Evaluate:

            Question: {q}
            Answer: {ans}
            Material: {context}

            Give:
            - Score (0-10)
            - What is correct
            - What is missing
            - Correct answer
            """

            res = model.invoke(eval_prompt)
            st.write(res.content)

        if st.button("Next Question"):
            st.session_state.pyq_idx = idx + 1
            st.rerun()
