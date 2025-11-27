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

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# ------------------------------------------------------
#                 ZEN STREAMLIT THEME (FINAL)
# ------------------------------------------------------

st.set_page_config(
    page_title="EduMind AI",
    page_icon="🕊️",
    layout="wide"
)

ZEN_CSS = """
<style>

/* Full reset */
html, body {
    background: linear-gradient(135deg, #eef0f3 0%, #e8e9ec 40%, #e1e2e4 100%) !important;
    height: 100%;
    width: 100%;
}

/* Force override Streamlit wrappers */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f2f3f5 0%, #e5e6e9 40%, #e0e1e4 100%) !important;
}

/* Remove hidden white overlay containers */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="StyledContainer"] { background: transparent !important; }

/* Main Content Glass Box */
.main .block-container {
    background: rgba(255, 255, 255, 0.55) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;

    max-width: 880px !important;
    padding: 3rem 2.5rem !important;
    margin: 3rem auto !important;

    box-shadow: 0 12px 45px rgba(0,0,0,0.08) !important;
}

/* Soft Header */
[data-testid="stHeader"] {
    background: rgba(255,255,255,0.3) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid rgba(0,0,0,0.08) !important;
}

/* Typography */
h1, h2, h3, h4 {
    color: #222 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

/* Buttons */
.stButton > button {
    border-radius: 14px !important;
    border: 1px solid #d4d4d6 !important;
    background-color: #efeff1 !important;
    color: #222 !important;

    padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #e3e3e5 !important;
    transform: translateY(-2px) !important;
}

/* Input Fields */
textarea, input, select, .stTextArea textarea {
    border-radius: 12px !important;
    background-color: #fafafa !important;
    border: 1px solid #d9d9de !important;
    padding: 10px 12px !important;
}

/* Radios / Select Labels */
.stRadio label, .stSelectbox label {
    font-weight: 500 !important;
    color: #333 !important;
}

/* Alerts */
.stAlert {
    border-radius: 14px !important;
    background-color: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(8px) !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.07) !important;
    margin: 2rem 0 !important;
}

/* Hide Footer */
footer, #MainMenu {visibility: hidden !important;}

</style>
"""
st.markdown(ZEN_CSS, unsafe_allow_html=True)


# ------------------------------------------------------
#   SIMPLE VECTOR STORE
# ------------------------------------------------------

class SimpleVectorStore:
    def __init__(self, embedding_function):
        self.embedding_fn = embedding_function
        self.vectors = []
        self.texts = []

    def add_documents(self, documents):
        for doc in documents:
            emb = self.embedding_fn.embed_query(doc.page_content)
            self.vectors.append(emb)
            self.texts.append(doc.page_content)

    def similarity_search(self, query, k=3):
        if not self.texts:
            return []

        qemb = self.embedding_fn.embed_query(query)
        scored = []

        for t, v in zip(self.texts, self.vectors):
            score = sum(a * b for a, b in zip(qemb, v))
            scored.append((score, t))

        scored.sort(reverse=True)
        top = scored[:k]
        return [Document(page_content=t) for (_, t) in top]


# ------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["Google_Search_API"]
GOOGLE_SEARCH_CX = st.secrets["Search_CX"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


# ------------------------------------------------------
# GOOGLE SEARCH API FUNCTION
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
# UNIQUE CHUNK SELECTOR
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
# DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    final = []
    for d in docs:
        chunks = splitter.split_text(d["page_content"])
        for c in chunks:
            final.append({"page_content": c})

    return final


def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]


# ------------------------------------------------------
# PDF LOADER
# ------------------------------------------------------

def load_pdf_auto(path: str):
    reader = PdfReader(path)
    has_text = any((p.extract_text() or "").strip() for p in reader.pages)

    if has_text:
        docs = PyPDFLoader(path).load()
        return [{"page_content": d.page_content} for d in docs]

    return [{"page_content": "This PDF does not contain selectable text."}]


# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------

st.title("Upload the PDF file")

uploaded_file = st.file_uploader("Upload your material", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("PDF Uploaded!")
    st.write("Processing… please wait ⏳")

    docs = load_pdf_auto(tmp_path)
    chunks = chunk_documents(docs)
    chunk_docs = to_document(chunks)

    vs = SimpleVectorStore(embeddings)
    vs.add_documents(chunk_docs)

    st.session_state.vector_store = vs
    st.session_state.chunks = chunks

    st.success("Embeddings generated & stored!")


# ------------------------------------------------------
# MODE SELECTOR
# ------------------------------------------------------

st.subheader("Choose Question Type")

question_mode = st.radio(
    "How do you want your questions?",
    ("Generated From the Notes", "PYQ")
)


# ------------------------------------------------------
# MODE 1 – GENERATED QUESTIONS
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
        if not st.session_state.chunks:
            st.warning("Upload a PDF first.")
        else:
            selector = UniqueChunkSelector(st.session_state.chunks)
            ch = selector.get_next_unique_chunk()

            st.session_state.current_chunk = ch["page_content"]

            prompt = f"""
            Generate 1 exam-style question strictly from this material:
            {ch["page_content"]}
            """

            out = model.invoke(prompt)
            st.session_state.generated_question = out.content
            st.session_state.user_ans = ""
            st.session_state.evaluation = None

    if st.session_state.generated_question:
        st.write("## Question:")
        st.write(st.session_state.generated_question)

        st.session_state.user_ans = st.text_area(
            "Write your answer...",
            value=st.session_state.user_ans
        )

        if st.button("Submit"):
            eval_prompt = f"""
            Evaluate the answer below:

            Question: {st.session_state.generated_question}
            User Answer: {st.session_state.user_ans}
            Material: {st.session_state.current_chunk}

            Provide:
            - Correctness 0–10
            - What is correct
            - Missing points
            - Ideal answer
            """

            response = model.invoke(eval_prompt)
            st.session_state.evaluation = response.content

    if st.session_state.evaluation:
        st.write("## Evaluation:")
        st.write(st.session_state.evaluation)


# ------------------------------------------------------
# MODE 2 – PYQ
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
    if "pyq_evaluation" not in st.session_state:
        st.session_state.pyq_evaluation = None

    st.info("🎯 PYQ Finder")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox(
            "Course",
            ["B.Tech", "MCA", "B.Pharm", "BBA", "BCA", "MBA"]
        )
        year = st.text_input("Year (optional)", placeholder="e.g., 2022")

    with col2:
        subject = st.text_input("Subject Code or Name", placeholder="e.g. KCS301")

    if st.button("Search"):
        if not subject:
            st.error("Enter a subject code/name.")
        else:
            with st.spinner("Searching…"):

                query = f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf'

                data = google_search(query, num_results=5)
                urls = [i["link"] for i in data.get("items", []) if "link" in i]

                if not urls:
                    st.warning("No PDFs found.")
                    st.stop()

                all_text = ""
                headers = {"User-Agent": "Mozilla/5.0"}

                files_found = 0

                for link in urls:
                    try:
                        if link.lower().endswith(".pdf"):
                            r = requests.get(link, headers=headers, timeout=8)
                            with open("temp_pyq.pdf", "wb") as f:
                                f.write(r.content)

                            reader = PdfReader("temp_pyq.pdf")
                            t = "".join((p.extract_text() or "") for p in reader.pages)

                            if t.strip():
                                all_text += f"\n--- SOURCE: {link} ---\n" + t
                                files_found += 1
                    except:
                        pass

                if not all_text:
                    st.error("Could not extract text.")
                    st.stop()

                extraction_prompt = f"""
                Extract ONLY exam questions from the text below.
                Return Python list: ["Q1", "Q2", ...]

                Text:
                {all_text[:30000]}
                """

                resp = model.invoke(extraction_prompt)
                clean = resp.content.replace("```", "").strip()

                try:
                    arr = ast.literal_eval(clean)
                except:
                    try:
                        arr = json.loads(clean)
                    except:
                        st.error("Error parsing questions.")
                        st.stop()

                if isinstance(arr, list) and arr:
                    st.session_state.pyq_list = arr
                    st.session_state.pyq_index = 0
                    st.success(f"Found {len(arr)} questions.")
                    st.rerun()
                else:
                    st.warning("No valid questions found.")

    # Display Questions
    if st.session_state.pyq_list:

        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        st.divider()
        st.info(f"## Question {idx + 1} of {total}")

        q = st.session_state.pyq_list[idx]
        st.write(q)
        st.session_state.current_question = q

        st.session_state.user_answer = st.text_area(
            "Your answer:",
            value=st.session_state.user_answer
        )

        if st.button("Evaluate Answer"):

            if not st.session_state.user_answer:
                st.warning("Type an answer first.")
            elif st.session_state.vector_store is None:
                st.error("Upload your notes PDF first.")
            else:

                with st.spinner("Grading…"):
                    retrieved = st.session_state.vector_store.similarity_search(q, k=3)
                    ctx = "\n\n".join(d.page_content for d in retrieved)
                    st.session_state.related_chunks = ctx

                    eval_prompt = f"""
                    Evaluate answer:

                    Question: {q}
                    User Answer: {st.session_state.user_answer}
                    Notes: {ctx}

                    Give:
                    - Score (0-10)
                    - Correct parts
                    - Missing parts
                    - Ideal answer
                    """

                    resp = model.invoke(eval_prompt)
                    st.session_state.pyq_evaluation = resp.content

        if st.session_state.pyq_evaluation:
            st.write("## Evaluation:")
            st.write(st.session_state.pyq_evaluation)

        if st.button("NEXT QUESTION"):
            st.session_state.user_answer = ""
            st.session_state.pyq_evaluation = None
            st.session_state.related_chunks = ""
            st.session_state.pyq_index += 1
            st.rerun()
