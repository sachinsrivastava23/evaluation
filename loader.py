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

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]

# Embeddings Model
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

# ------------------------------------------------------
#            DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len
    )
    final = []

    for d in docs:
        text = re.sub(r"\s+", " ", d["page_content"])
        for c in splitter.split_text(text):
            if c.strip():
                final.append(Document(page_content=c))
    return final

# ------------------------------------------------------
#        RAW CHROMA CLIENT — LATEST API
# ------------------------------------------------------

from chromadb import PersistentClient

CHROMA_PATH = "chroma_db"
client = PersistentClient(path=CHROMA_PATH)

# ⛔ Do NOT pass embedding_function
collection = client.get_or_create_collection(
    "pdf_chunks",
    metadata={"hnsw:space": "cosine"}
)

def add_to_chroma(docs):
    for i, doc in enumerate(docs):
        emb = embedder.embed_documents([doc.page_content])[0]

        collection.add(
            ids=[f"doc_{i}"],
            documents=[doc.page_content],
            embeddings=[emb],
        )
    client.persist()

def chroma_search(query, k=3):
    q_emb = embedder.embed_query(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    return res["documents"][0] if res["documents"] else []

# ------------------------------------------------------
#              PDF LOADING
# ------------------------------------------------------

def load_pdf(path):
    reader = PdfReader(path)
    docs = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            docs.append({"page_content": t})
    return docs

# ------------------------------------------------------
#                STREAMLIT UI
# ------------------------------------------------------

st.title("📘 Smart Study Companion (Stable Version)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded!")
    st.write("Processing… ⏳")

    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)

    add_to_chroma(chunks)

    st.success("Embeddings stored successfully!")

# ------------------------------------------------------
#              QUESTION GENERATION
# ------------------------------------------------------

mode = st.radio("Choose Mode", ["Generated From Notes", "PYQ"])

# ------------------------- MODE 1 -----------------------

if mode == "Generated From Notes":

    if st.button("Generate Question"):
        if not chunks:
            st.error("Upload a PDF first.")
        else:
            ch = random.choice(chunks).page_content

            prompt = f"""
            Create 1 exam-style question from the text below:
            {ch}
            """

            q = model.invoke(prompt).content
            st.write("### Question:")
            st.write(q)

            ans = st.text_area("Your Answer:")

            if st.button("Evaluate"):
                retrieved = chroma_search(q, k=3)
                ctx = "\n".join(retrieved)

                eval_prompt = f"""
                Question: {q}
                Answer: {ans}
                Notes: {ctx}

                Evaluate:
                - Score /10
                - Correct parts
                - Missing parts
                - Improved answer
                """

                res = model.invoke(eval_prompt).content
                st.write(res)

# ------------------------- MODE 2 -----------------------

if mode == "PYQ":

    subject = st.text_input("Subject Code or Name")
    year = st.text_input("Year (optional)")

    if st.button("Search PYQs"):
        query = f'site:abesit.in "{subject}" {year} filetype:pdf'
        data = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_SEARCH_API_KEY,
                "cx": GOOGLE_SEARCH_CX,
                "q": query
            }
        ).json()

        urls = [i["link"] for i in data.get("items", []) if "link" in i]

        text = ""
        for link in urls:
            if link.endswith(".pdf"):
                try:
                    r = requests.get(link, timeout=8)
                    with open("temp.pdf", "wb") as f:
                        f.write(r.content)

                    reader = PdfReader("temp.pdf")
                    for p in reader.pages:
                        if p.extract_text():
                            text += p.extract_text()
                except:
                    pass

        extract_prompt = f"""
        Extract ONLY exam questions.
        Text:
        {text[:30000]}

        Return: ["Q1", "Q2"]
        """

        clean = model.invoke(extract_prompt).content.replace("```", "")
        try:
            pyqs = ast.literal_eval(clean)
        except:
            pyqs = []

        if pyqs:
            st.session_state.pyqs = pyqs
            st.session_state.index = 0
            st.success(f"Found {len(pyqs)} questions.")
            st.rerun()

    if "pyqs" in st.session_state:
        idx = st.session_state.index
        q = st.session_state.pyqs[idx]

        st.write(f"### Question {idx+1}:")
        st.write(q)

        ans = st.text_area("Your Answer:")

        if st.button("Evaluate PYQ"):
            retrieved = chroma_search(q, k=3)
            ctx = "\n".join(retrieved)

            eval_prompt = f"""
            Question: {q}
            Answer: {ans}
            Notes: {ctx}

            Evaluate:
            - Score /10
            - Correct points
            - Missing points
            - Best answer
            """

            res = model.invoke(eval_prompt).content
            st.write(res)

        if st.button("Next"):
            st.session_state.index += 1
            st.rerun()
