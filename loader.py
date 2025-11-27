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

# VECTOR DB (QDRANT CLOUD)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant

# ------------------------------------------------------
#                  INITIAL SETUP
# ------------------------------------------------------

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

# ------------------------------------------------------
#                GOOGLE SEARCH API
# ------------------------------------------------------

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": num_results,
    }
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {}

# ------------------------------------------------------
#             UNIQUE CHUNK SELECTOR
# ------------------------------------------------------

class UniqueChunkSelector:
    def __init__(self, all_chunks):
        self.chunk = all_chunks
        self.total_count = len(all_chunks)
        self.used = set()

    def get_next(self):
        remaining = list(set(range(self.total_count)) - self.used)
        if not remaining:
            return "all chunks used"
        idx = random.choice(remaining)
        self.used.add(idx)
        return self.chunk[idx]

# ------------------------------------------------------
#           EMBEDDINGS MODEL
# ------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# ------------------------------------------------------
#     PDF TEXT EXTRACTOR (NO OCR)
# ------------------------------------------------------

def load_pdf_auto(file_path):
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            docs = PyPDFLoader(file_path).load()
            return [{"page_content": d.page_content} for d in docs]

    return [{"page_content": "This PDF has no extractable text."}]

# ------------------------------------------------------
#       SPLIT TEXT INTO CHUNKS
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    all_chunks = []

    for d in docs:
        parts = splitter.split_text(d["page_content"])
        for p in parts:
            all_chunks.append({"page_content": p})

    return all_chunks

def to_documents(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#             STREAMLIT UI
# ------------------------------------------------------

st.title("📘 Smart Study Companion (Qdrant Version)")

uploaded_file = st.file_uploader("Upload PDF Notes", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded! Processing...")

    docs = load_pdf_auto(pdf_path)
    chunk = chunk_documents(docs)
    chunk_docs = to_documents(chunk)

    COLLECTION_NAME = "user_pdf_chunks"

    # 🔥 Upload to Qdrant Cloud
    vector_store = Qdrant.from_documents(
        documents=chunk_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    st.success("Embeddings uploaded to Qdrant Cloud!")

# ------------------------------------------------------
#          QUESTION MODE SELECTOR
# ------------------------------------------------------

st.subheader("Choose Question Type")
choice = st.radio("", ["Generated From Notes", "PYQ"])

# ------------------------------------------------------
#         MODE 1 — GENERATED FROM NOTES
# ------------------------------------------------------

if choice == "Generated From Notes":

    if "generated_question" not in st.session_state:
        st.session_state.generated_question = None
    if "current_chunk" not in st.session_state:
        st.session_state.current_chunk = None
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""
    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    # Generate question
    if st.button("Generate Question"):
        selector = UniqueChunkSelector(chunk)
        chunk_text = selector.get_next()
        st.session_state.current_chunk = chunk_text

        prompt = f"""
        Generate one exam-style question from the following material:

        {chunk_text}
        """

        out = model.invoke(prompt)
        st.session_state.generated_question = out.content

    # Show question
    if st.session_state.generated_question:
        st.write("### Question:")
        st.write(st.session_state.generated_question)

        st.session_state.user_answer = st.text_area(
            "Your Answer:", value=st.session_state.user_answer
        )

        if st.button("Evaluate Answer"):
            eval_prompt = f"""
            Evaluate the answer strictly from the material.

            Question: {st.session_state.generated_question}
            User Answer: {st.session_state.user_answer}
            Study Material: {st.session_state.current_chunk}

            Return:
            - Score (0-10)
            - What is correct
            - What is missing
            - Correct answer
            """

            result = model.invoke(eval_prompt)
            st.session_state.evaluation = result.content

    if st.session_state.evaluation:
        st.write("### Evaluation:")
        st.write(st.session_state.evaluation)

# ------------------------------------------------------
#                MODE 2 — PYQ SEARCH
# ------------------------------------------------------

if choice == "PYQ":

    if "pyq_list" not in st.session_state:
        st.session_state.pyq_list = []
    if "pyq_index" not in st.session_state:
        st.session_state.pyq_index = 0

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox("Course", ["B.Tech", "MCA", "BCA", "MBA", "BBA"])
        year = st.text_input("Year (optional)")

    with col2:
        subject = st.text_input("Subject / Code")
        st.caption("Tip: Use subject code like KCS301")

    if st.button("Search PYQ"):
        if not subject:
            st.error("Enter subject")
            st.stop()

        query = f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf"'
        data = google_search(query)

        urls = [i["link"] for i in data.get("items", []) if "link" in i]

        if not urls:
            st.error("No PYQs found.")
            st.stop()

        all_text = ""
        for url in urls:
            try:
                if url.endswith(".pdf"):
                    r = requests.get(url)
                    with open("temp.pdf", "wb") as f:
                        f.write(r.content)

                    reader = PdfReader("temp.pdf")
                    for pg in reader.pages:
                        t = pg.extract_text()
                        if t:
                            all_text += t + "\n"
            except:
                pass

        extract_prompt = f"""
        Extract ONLY exam questions from this text.
        Output as a Python list of strings.

        {all_text[:30000]}
        """

        res = model.invoke(extract_prompt).content
        clean = re.sub(r"```.*?```", "", res, flags=re.DOTALL).strip()

        try:
            pyqs = ast.literal_eval(clean)
        except:
            pyqs = []

        st.session_state.pyq_list = pyqs
        st.session_state.pyq_index = 0

        st.success(f"Found {len(pyqs)} questions!")

    if st.session_state.pyq_list:
        idx = st.session_state.pyq_index
        q = st.session_state.pyq_list[idx]

        st.write(f"### Question {idx+1}")
        st.write(q)

        user_ans = st.text_area("Your Answer:")

        if st.button("Evaluate PYQ"):
            vector_store = Qdrant(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name="user_pdf_chunks",
                embeddings=embeddings,
            )

            retrieved = vector_store.similarity_search(q, k=3)
            context = "\n\n".join([d.page_content for d in retrieved])

            prompt = f"""
            Evaluate based on the following notes:

            Notes:
            {context}

            User Answer:
            {user_ans}

            Provide:
            - Correctness score 0–10
            - What is correct
            - What is missing
            - Full correct answer
            """

            result = model.invoke(prompt).content
            st.write("### Evaluation:")
            st.write(result)

        if st.button("Next"):
            st.session_state.pyq_index += 1
            st.rerun()
