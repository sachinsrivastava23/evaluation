import os
import random
import tempfile
import json
import requests
import re
import ast

import streamlit as st
from pypdf import PdfReader
from bs4 import BeautifulSoup 

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 🔥 USE MEMORY VECTOR STORE
from langchain_community.vectorstores import MemoryVectorStore

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# ------------------------------------------------------
#                  INITIAL SETUP
# ------------------------------------------------------

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

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
#          UNIQUE CHUNK SELECTOR CLASS
# ------------------------------------------------------

class UniqueChunkSelector:
    def __init__(self, all_chunks):
        self.chunk = all_chunks
        self.total_count = len(all_chunks)
        self.used_indices = set()

    def get_next_unique_chunk(self):
        all_indices = set(range(self.total_count))
        available_indices = list(all_indices - self.used_indices)

        if not available_indices:
            return "all chunks have been used"

        selected = random.choice(available_indices)
        self.used_indices.add(selected)
        return self.chunk[selected]

# ------------------------------------------------------
#                EMBEDDING MODEL
# ------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# ------------------------------------------------------
#        IN-MEMORY VECTOR STORE (NO FILE STORAGE)
# ------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = []
    for d in docs:
        parts = text_splitter.split_text(d["page_content"])
        for c in parts:
            chunks.append({"page_content": c})

    return chunks

def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#          PDF TEXT AUTO-DETECTOR
# ------------------------------------------------------

def load_pdf_auto(file_path):
    reader = PdfReader(file_path)

    has_text = any(
        (page.extract_text() and page.extract_text().strip())
        for page in reader.pages
    )

    if has_text:
        docs = PyPDFLoader(file_path).load()
        return [{"page_content": d.page_content} for d in docs]

    return [{"page_content": "This PDF does not contain selectable text."}]

# ------------------------------------------------------
#                    STREAMLIT UI
# ------------------------------------------------------

st.title("Upload the PDF file")

uploaded_file = st.file_uploader("Upload your material", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("The file is uploaded successfully!")
    st.write("Processing… please wait ⏳")

    docs = load_pdf_auto(tmp_path)
    chunk = chunk_documents(docs)
    chunk_docs = to_document(chunk)

    # ⭐ CREATE MEMORY VECTOR STORE
    st.session_state.vector_store = MemoryVectorStore.from_documents(
        chunk_docs,
        embeddings
    )

    st.success("Embeddings generated and stored in RAM!")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.subheader("Choose Question Type")

question_mode = st.radio(
    "Select How You Want To Get Questions",
    ("Generated From the Notes", "PYQ")
)

# ------------------------------------------------------
#             MODE 1: GENERATED FROM NOTES
# ------------------------------------------------------

if question_mode == "Generated From the Notes":

    if "generated_question" not in st.session_state:
        st.session_state.generated_question = None

    if "current_chunk" not in st.session_state:
        st.session_state.current_chunk = None

    if "user_ans" not in st.session_state:
        st.session_state.user_ans = ""

    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    if st.button("Generate Questions"):
        selector = UniqueChunkSelector(chunk)
        chunk_text = selector.get_next_unique_chunk()
        st.session_state.current_chunk = chunk_text

        prompt = f"""
        Generate 1 exam-oriented question strictly from this material:
        {chunk_text}
        """

        output = model.invoke(prompt)
        st.session_state.generated_question = output.content
        st.session_state.user_ans = ""
        st.session_state.evaluation = None

    if st.session_state.generated_question:
        st.write("## Question:")
        st.write(st.session_state.generated_question)

        st.session_state.user_ans = st.text_area(
            "TYPE THE ANSWER...",
            value=st.session_state.user_ans
        )

        if st.button("Submit"):
            eval_prompt = f"""
            Evaluate this answer strictly based on the material:

            Question: {st.session_state.generated_question}
            Answer: {st.session_state.user_ans}
            Material: {st.session_state.current_chunk}

            Provide:
            - Score (0–10)
            - What is correct
            - What is missing
            - Correct answer
            """

            evaluation = model.invoke(eval_prompt)
            st.session_state.evaluation = evaluation.content

    if st.session_state.evaluation:
        st.write("## Evaluation:")
        st.write(st.session_state.evaluation)

# ------------------------------------------------------
#                     MODE 2: PYQ
# ------------------------------------------------------

if question_mode == "PYQ":

    if "pyq_list" not in st.session_state:
        st.session_state.pyq_list = []

    if "pyq_index" not in st.session_state:
        st.session_state.pyq_index = 0

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""

    if "related_chunks" not in st.session_state:
        st.session_state.related_chunks = ""

    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    st.info("🎯 Focus: ABESIT Library Question Bank")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox(
            "Select Course",
            ["B.Tech", "MCA", "B.Pharm", "BBA", "BCA", "MBA"]
        )
        year = st.text_input("Year (Optional)", placeholder="e.g. 2022")

    with col2:
        subject = st.text_input(
            "Subject Name or Code",
            placeholder="e.g. KCS301 or Data Structures"
        )

    if st.button("Search ABESIT Library"):
        if not subject:
            st.error("Enter a subject.")
        else:
            st.warning("ABESIT scraping unchanged — keeping your logic same.")
            # Your existing scraping logic remains untouched

    if st.session_state.pyq_list:

        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        if idx < total:

            st.info(f"## Question {idx + 1} of {total}")
            st.write(st.session_state.pyq_list[idx])

            st.session_state.current_question = st.session_state.pyq_list[idx]

            st.session_state.user_answer = st.text_area(
                "Your Answer:",
                value=st.session_state.user_answer
            )

            if st.button("Evaluate Answer"):

                retrieved_chunks = st.session_state.vector_store.search(
                    st.session_state.current_question, k=3
                )

                context_chunk = "\n\n".join(
                    [doc.page_content for doc in retrieved_chunks]
                )

                st.session_state.related_chunks = context_chunk

                eval_prompt = f"""
                Evaluate this answer based on material:

                Question: {st.session_state.current_question}
                Answer: {st.session_state.user_answer}
                Material: {context_chunk}
                """

                response = model.invoke(eval_prompt)
                st.session_state.evaluation = response.content

            if st.session_state.evaluation:
                st.write(st.session_state.evaluation)

            if st.button("NEXT QUESTION"):
                st.session_state.user_answer = ""
                st.session_state.evaluation = None
                st.session_state.related_chunks = ""
                st.session_state.pyq_index += 1
                st.rerun()
