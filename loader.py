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
from langchain_community.vectorstores import FAISS  # ← REPLACED CHROMA WITH FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# ------------------------------------------------------
#                  INITIAL SETUP
# ------------------------------------------------------

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Load secrets
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ------------------------------------------------------
#            GOOGLE SEARCH API FUNCTION
# ------------------------------------------------------

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_SEARCH_API_KEY, "cx": GOOGLE_SEARCH_CX, "q": query, "num": num_results}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except:
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
        available = list(set(range(self.total_count)) - self.used_indices)
        if not available:
            return "all chunks have been used"
        idx = random.choice(available)
        self.used_indices.add(idx)
        return self.chunk[idx]

# ------------------------------------------------------
#                EMBEDDING MODEL
# ------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# ------------------------------------------------------
#               FAISS VECTOR STORE (NEW)
# ------------------------------------------------------

faiss_store = None   # will hold FAISS index later

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    out = []
    for d in docs:
        pieces = splitter.split_text(d["page_content"])
        for c in pieces:
            out.append({"page_content": c})
    return out

def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#          PDF TEXT AUTO-DETECTOR (NO OCR)
# ------------------------------------------------------

def load_pdf_auto(path):
    reader = PdfReader(path)
    for page in reader.pages:
        txt = page.extract_text()
        if txt and txt.strip():
            docs = PyPDFLoader(path).load()
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
        pdf_path = tmp.name

    st.success("File uploaded!")
    st.write("Processing…")

    docs = load_pdf_auto(pdf_path)
    chunks = chunk_documents(docs)
    chunk_docs = to_document(chunks)

    # Build FAISS index here (in RAM)
    faiss_store = FAISS.from_documents(chunk_docs, embeddings)

    st.success("Embeddings generated (FAISS index created)!")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.subheader("Choose Question Type")
question_mode = st.radio("Select Mode:", ("Generated From the Notes", "PYQ"))

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
        selector = UniqueChunkSelector(chunks)
        picked = selector.get_next_unique_chunk()
        st.session_state.current_chunk = picked

        prompt = f"""
        Generate 1 exam-oriented question ONLY from this material:
        {picked}
        """
        ans = model.invoke(prompt)
        st.session_state.generated_question = ans.content
        st.session_state.user_ans = ""
        st.session_state.evaluation = None

    if st.session_state.generated_question:
        st.write("### Question:")
        st.write(st.session_state.generated_question)

        st.session_state.user_ans = st.text_area("TYPE THE ANSWER...", value=st.session_state.user_ans)

        if st.button("Submit"):
            prompt = f"""
            Evaluate the answer ONLY using this study material:

            Question: {st.session_state.generated_question}
            User Answer: {st.session_state.user_ans}
            Material: {st.session_state.current_chunk}

            Give:
            - Score (0–10)
            - What is correct
            - What is missing
            - Correct answer
            """
            out = model.invoke(prompt)
            st.session_state.evaluation = out.content

    if st.session_state.evaluation:
        st.write("### Evaluation:")
        st.write(st.session_state.evaluation)

# ------------------------------------------------------
#                     MODE 2: PYQ
# ------------------------------------------------------

if question_mode == "PYQ":

    if "pyq_list" not in st.session_state:
        st.session_state.pyq_list = []

    if "pyq_index" not in st.session_state:
        st.session_state.pyq_index = 0

    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""

    if "related_chunks" not in st.session_state:
        st.session_state.related_chunks = ""

    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    st.info("🎯 ABESIT Library Question Bank")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox("Select Course", ["B.Tech","MCA","B.Pharm","BBA","BCA","MBA"])
        year = st.text_input("Year", placeholder="e.g. 2022")

    with col2:
        subject = st.text_input("Subject Name or Code")
        st.caption("Use subject codes like KCS301 for best results.")

    if st.button("Search ABESIT Library"):

        if not subject:
            st.error("Please enter subject.")
        else:
            query = f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf'
            data = google_search(query)

            urls = [item["link"] for item in data.get("items", []) if "link" in item]

            if not urls:
                st.warning("No PDFs found.")
                st.stop()

            all_text = ""
            for link in urls:
                if link.lower().endswith(".pdf"):
                    try:
                        r = requests.get(link, timeout=8)
                        with open("pyq.pdf", "wb") as f:
                            f.write(r.content)

                        reader = PdfReader("pyq.pdf")
                        temp = ""
                        for page in reader.pages:
                            t = page.extract_text()
                            if t:
                                temp += t + "\n"

                        if temp.strip():
                            all_text += temp + "\n"
                    except:
                        pass

            extract_prompt = f"""
            Extract ONLY exam questions from this text:

            {all_text[:30000]}

            Output only a Python list: ["Q1", "Q2", ...]
            """

            res = model.invoke(extract_prompt)
            clean = re.sub(r"```.*?```", "", res.content, flags=re.DOTALL).strip()

            try:
                arr = ast.literal_eval(clean)
            except:
                arr = []

            if arr:
                st.session_state.pyq_list = arr
                st.session_state.pyq_index = 0
                st.success(f"Found {len(arr)} questions!")
                st.rerun()

    if st.session_state.pyq_list:

        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        if idx < total:
            st.write(f"### Question {idx+1}/{total}")
            st.write(st.session_state.pyq_list[idx])

            st.session_state.current_question = st.session_state.pyq_list[idx]

            st.session_state.user_answer = st.text_area("Your Answer:", value=st.session_state.user_answer)

            if st.button("Evaluate Answer"):

                # RAG retrieval using FAISS
                retrieved = faiss_store.similarity_search(st.session_state.current_question, k=3)
                context = "\n\n".join(d.page_content for d in retrieved)
                st.session_state.related_chunks = context

                prompt = f"""
                Evaluate the answer using this material:

                Question: {st.session_state.current_question}
                User Answer: {st.session_state.user_answer}
                Material: {context}

                Give:
                - Score (0–10)
                - What is correct
                - What is missing
                - Correct answer
                """

                result = model.invoke(prompt)
                st.session_state.evaluation = result.content

            if st.session_state.evaluation:
                st.write(st.session_state.evaluation)

            if st.button("NEXT QUESTION"):
                st.session_state.user_answer = ""
                st.session_state.evaluation = None
                st.session_state.related_chunks = ""
                st.session_state.pyq_index += 1
                st.rerun()

        else:
            st.warning("That was the last question!")
            if st.button("Restart"):
                st.session_state.pyq_index = 0
                st.rerun()
