import os
import random
import tempfile
import json
import requests
import re
import ast

import streamlit as st
from pypdf import PdfReader
from bs4 import BeautifulSoup  # still here if you want later

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# ------------------------------------------------------
#                 ZEN STREAMLIT THEME
# ------------------------------------------------------

st.set_page_config(
    page_title="EduMind AI",
    page_icon="🕊️",
    layout="centered"
)

ZEN_CSS = """
<style>

.stApp {
    background: radial-gradient(circle at top, #F7F7F8 0%, #F1F2F4 40%, #ECEDEF 100%);
    color: #222427;
}

/* Center content */
.main .block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

header, [data-testid="stHeader"] {
    background: transparent;
    border-bottom: none;
}

[data-testid="stVerticalBlock"] > div {
    border-radius: 16px !important;
    padding: 0.5rem 0.7rem;
}

h1, h2, h3, h4 {
    font-weight: 500;
    letter-spacing: 0.02em;
    color: #222427;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid #D0D2D6;
    background-color: #E4E5E7;
    color: #222427 !important;
    padding: 0.45rem 1.3rem;
    font-weight: 400;
    transition: all 180ms ease-out;
}

.stButton > button:hover {
    background-color: #D7D8DC;
    border-color: #C2C4C9;
    transform: translateY(-1px);
}

textarea, input, select {
    border-radius: 12px !important;
}

.stTextArea textarea {
    border-radius: 12px !important;
    background-color: #FAFAFA !important;
    border: 1px solid #D8D9DD !important;
}

.stSelectbox div[data-baseweb="select"],
.stRadio label,
.stTextInput input {
    background-color: #FAFAFA !important;
    border-radius: 12px !important;
}

.stAlert {
    border-radius: 14px !important;
}

footer {visibility: hidden;}
</style>
"""

st.markdown(ZEN_CSS, unsafe_allow_html=True)

# ------------------------------------------------------
#   SIMPLE VECTOR STORE (RAM ONLY)
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
#                  INITIAL SETUP
# ------------------------------------------------------

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_CX = st.secrets["GOOGLE_SEARCH_CX"]

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

        selected_index = random.choice(available_indices)
        self.used_indices.add(selected_index)
        return self.chunk[selected_index]

# ------------------------------------------------------
#              DOCUMENT CHUNKING
# ------------------------------------------------------

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    final_chunk = []
    for d in docs:
        chunks = text_splitter.split_text(d["page_content"])
        for c in chunks:
            final_chunk.append({"page_content": c})

    return final_chunk

def to_document(chunks):
    return [Document(page_content=c["page_content"]) for c in chunks]

# ------------------------------------------------------
#       PDF TEXT AUTO-DETECTOR
# ------------------------------------------------------

def load_pdf_auto(file_path: str):
    reader = PdfReader(file_path)
    has_text = False

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            has_text = True
            break

    if has_text:
        docs = PyPDFLoader(file_path).load()
        return [{"page_content": d.page_content} for d in docs]

    return [{"page_content": "This PDF does not contain selectable text. Unsupported format."}]

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
    chunks = chunk_documents(docs)
    chunk_docs = to_document(chunks)

    vs = SimpleVectorStore(embeddings)
    vs.add_documents(chunk_docs)

    st.session_state.vector_store = vs
    st.session_state.chunks = chunks

    st.success("Embeddings generated and stored in memory for this session!")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.subheader("Choose Question Type")

question_mode = st.radio(
    "Select How You Want To Get Questions",
    ("Generated From the Notes", "PYQ")
)

# ------------------------------------------------------
#     MODE 1 — GENERATED FROM NOTES
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
            st.warning("Please upload a PDF first.")
        else:
            selector = UniqueChunkSelector(st.session_state.chunks)
            chunk_obj = selector.get_next_unique_chunk()
            chunk_text = chunk_obj["page_content"]
            st.session_state.current_chunk = chunk_text

            prompt = f"""
            Generate 1 exam-oriented question strictly from this material.
            No extra topics. No MCQs. No references.

            Study Material:
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
            evaluation_prompt = f"""
            Evaluate this answer strictly based on the material:

            Question: {st.session_state.generated_question}
            User Answer: {st.session_state.user_ans}
            Study Material: {st.session_state.current_chunk}

            Provide:
            - Correctness (0-10)
            - What is correct
            - What is missing
            - Correct answer
            """

            evaluation = model.invoke(evaluation_prompt)
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

    if "pyq_evaluation" not in st.session_state:
        st.session_state.pyq_evaluation = None

    st.info("🎯 Focus: PYQ Question Bank")

    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox(
            "Select Course",
            ["B.Tech", "MCA", "B.Pharm", "BBA", "BCA", "MBA"],
            key="pyq_course"
        )
        year = st.text_input("Year (Optional)", placeholder="e.g. 2022", key="pyq_year")

    with col2:
        subject = st.text_input(
            "Subject Name or Code",
            placeholder="e.g. KCS301 or Data Structures",
            key="pyq_sub"
        )
        st.caption("Tip: Subject Codes (like KCS-301) work best.")

    if st.button("Search"):

        if not subject:
            st.error("Please enter a Subject Name or Code.")
        else:
            with st.spinner(f"Scanning for {course} {subject}..."):

                search_query = (
                    f'site:abesit.in "{course}" "{subject}" {year} filetype:pdf'
                )

                try:
                    google_json = google_search(search_query, num_results=5)

                    urls = []
                    for item in google_json.get("items", []):
                        if "link" in item:
                            urls.append(item["link"])

                    if not urls:
                        st.warning(f"No PDFs found on ABESIT for '{subject}'.")
                        st.info("Try searching by the Subject Code (e.g., 'KCS-301').")
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
                                file_text = ""

                                for page in reader.pages:
                                    t = page.extract_text()
                                    if t:
                                        file_text += t + "\n"

                                if file_text.strip():
                                    all_text += f"\n--- SOURCE: {link} ---\n" + file_text
                                    files_found += 1
                        except:
                            pass

                    if not all_text.strip():
                        st.error("Found files but couldn't extract text.")
                        st.stop()

                    extraction_prompt = f"""
                    You are extracting questions from the ABESIT Library Question Bank.

                    Target Subject: {subject}
                    Target Course: {course}

                    Task:
                    1. Read the text below.
                    2. Extract ONLY the exam questions.
                    3. Do NOT generate fake questions.
                    4. If unrelated, return [].

                    Output: Python list of strings: ["Question 1", "Question 2", ...]

                    Text:
                    {all_text[:30000]}
                    """

                    response = model.invoke(extraction_prompt)
                    clean_text = re.sub(
                        r"```[a-zA-Z]*", "", response.content
                    ).replace("```", "").strip()

                    try:
                        question_array = ast.literal_eval(clean_text)
                    except:
                        try:
                            question_array = json.loads(clean_text)
                        except:
                            st.error("Error parsing extracted questions.")
                            st.stop()

                    if isinstance(question_array, list) and len(question_array) > 0:
                        st.session_state.pyq_list = question_array
                        st.session_state.pyq_index = 0
                        st.success(f"Found {len(question_array)} questions from {files_found} ABESIT paper(s).")
                        st.rerun()
                    else:
                        st.warning("No clear questions found.")

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.pyq_list:

        st.divider()
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

                if not st.session_state.user_answer:
                    st.warning("Please write an answer first.")
                elif st.session_state.vector_store is None:
                    st.error("Upload your notes PDF first so I can use them for evaluation.")
                else:
                    with st.spinner("Retrieving notes & grading..."):

                        retrieved_chunks = st.session_state.vector_store.similarity_search(
                            st.session_state.current_question, k=3
                        )

                        context_chunk = "\n\n".join(
                            [doc.page_content for doc in retrieved_chunks]
                        )

                        st.session_state.related_chunks = context_chunk

                        eval_prompt = f"""
                        Evaluate this answer strictly based on the material.
                        If the material is not enough, then only add missing info yourself.

                        Question: {st.session_state.current_question}
                        User Answer: {st.session_state.user_answer}
                        Study Material: {st.session_state.related_chunks}

                        Provide:
                        - Correctness (0-10)
                        - What is correct
                        - What is missing
                        - Correct answer
                        """

                        response = model.invoke(eval_prompt)
                        st.session_state.pyq_evaluation = response.content

            if st.session_state.pyq_evaluation:
                st.write("## Evaluation:")
                st.write(st.session_state.pyq_evaluation)

            if st.button("NEXT QUESTION"):
                st.session_state.user_answer = ""
                st.session_state.pyq_evaluation = None
                st.session_state.related_chunks = ""
                st.session_state.pyq_index += 1
                st.rerun()

        else:
            st.warning("That was the last question!")
            if st.button("Restart"):
                st.session_state.pyq_index = 0
                st.rerun()
