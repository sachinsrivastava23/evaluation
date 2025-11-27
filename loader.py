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
#   PAGE CONFIG & ZEN CSS (VISUALS ONLY)
# ------------------------------------------------------
st.set_page_config(
    page_title="Study Focus", 
    layout="wide", 
    page_icon="🌿"
)

st.markdown("""
<style>
    /* 1. Main Background - Soft Off-White */
    .stApp {
        background-color: #F9FAFB;
        color: #4A5568;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* 2. Typography - Clean, Dark Grey, No Harsh Black */
    h1, h2, h3 {
        color: #2D3748;
        font-weight: 300; /* Light/Thin font weight */
        letter-spacing: -0.5px;
    }
    
    h1 {
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 3rem;
        font-size: 2.2rem;
    }

    /* 3. Containers/Cards - White with very subtle shadow */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #FFFFFF;
        border: 1px solid #F0F0F0;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02); /* Extremely subtle lift */
    }

    /* 4. Inputs - Minimal Borders */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        color: #4A5568;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #CBD5E0;
        box-shadow: none; /* Removed harsh glow */
    }

    /* 5. Buttons - Soft, Muted, Rounded (Pill shape) */
    .stButton > button {
        background-color: #F7FAFC;
        color: #4A5568;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #EDF2F7;
        border-color: #CBD5E0;
        color: #2D3748;
    }
    
    /* Primary Action Button Style (Optional override for emphasis if needed, kept subtle) */
    .stButton > button:active {
        background-color: #E2E8F0;
    }

    /* 6. Alerts/Messages - Pastel tones */
    .stSuccess {
        background-color: #F0FFF4;
        color: #2F855A;
        border: none;
    }
    .stInfo {
        background-color: #EBF8FF;
        color: #2B6CB0;
        border: none;
    }
    .stWarning {
        background-color: #FFFFF0;
        color: #C05621;
        border: none;
    }
    
    /* Hide default header/footer for cleaner look */
    header {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
#   SIMPLE VECTOR STORE (RAM ONLY, NO EXTERNAL DB)
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
            # cosine-like score by dot product (good enough for now)
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

# keep vector store + chunks in session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ------------------------------------------------------
#           GOOGLE SEARCH API FUNCTION
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
#         UNIQUE CHUNK SELECTOR CLASS
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
#         PDF TEXT AUTO-DETECTOR (NO OCR)
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

st.markdown("<h1>Study & Focus</h1>", unsafe_allow_html=True)

# Using a container with the new CSS for a clean card look
with st.container():
    st.markdown("### Upload Material")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            docs = load_pdf_auto(tmp_path)
            chunks = chunk_documents(docs)
            chunk_docs = to_document(chunks)

            # create simple RAM vector store
            vs = SimpleVectorStore(embeddings)
            vs.add_documents(chunk_docs)

            st.session_state.vector_store = vs
            st.session_state.chunks = chunks

        st.success("Material ready.")

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True) # Gentle spacing

with st.container():
    st.markdown("### Select Mode")
    question_mode = st.radio(
        "Mode",
        ("Generated From the Notes", "PYQ"),
        label_visibility="collapsed",
        horizontal=True
    )

# ------------------------------------------------------
#              MODE 1: GENERATED FROM NOTES
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

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_center, col_rest = st.columns([1, 2])
    with col_center:
        if st.button("Generate Question"):
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
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown(f"**Question:** {st.session_state.generated_question}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.session_state.user_ans = st.text_area(
                "Your answer...",
                value=st.session_state.user_ans,
                height=150,
                label_visibility="collapsed"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Submit Answer"):
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
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("#### Evaluation")
            st.write(st.session_state.evaluation)

# ------------------------------------------------------
#                       MODE 2: PYQ (ABESIT)
# ------------------------------------------------------

if question_mode == "PYQ":

    # State Setup
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

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Inputs
    with st.container():
        st.markdown("#### Search Parameters")
        col1, col2 = st.columns(2)

        with col1:
            course = st.selectbox(
                "Course",
                ["B.Tech", "MCA", "B.Pharm", "BBA", "BCA", "MBA"],
                key="pyq_course"
            )
            year = st.text_input("Year", placeholder="e.g. 2022", key="pyq_year")

        with col2:
            subject = st.text_input(
                "Subject Code/Name",
                placeholder="e.g. KCS301",
                key="pyq_sub"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Search Papers"):

            if not subject:
                st.error("Please enter a Subject Name or Code.")
            else:
                with st.spinner(f"Searching..."):

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
                            st.warning(f"No PDFs found.")
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
                            st.error("No text extracted.")
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
                                st.error("Parsing error.")
                                st.stop()

                        if isinstance(question_array, list) and len(question_array) > 0:
                            st.session_state.pyq_list = question_array
                            st.session_state.pyq_index = 0
                            st.success(f"Found {len(question_array)} questions.")
                            st.rerun()
                        else:
                            st.warning("No questions found.")

                    except Exception as e:
                        st.error(f"Error: {e}")

    # Display Questions
    if st.session_state.pyq_list:

        st.markdown("<br>", unsafe_allow_html=True)
        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        if idx < total:
            
            with st.container():
                st.caption(f"Question {idx + 1} / {total}")
                st.markdown(f"**{st.session_state.pyq_list[idx]}**")

                st.session_state.current_question = st.session_state.pyq_list[idx]

                st.markdown("<br>", unsafe_allow_html=True)
                st.session_state.user_answer = st.text_area(
                    "Your Answer",
                    value=st.session_state.user_answer,
                    height=150,
                    label_visibility="collapsed"
                )

                st.markdown("<br>", unsafe_allow_html=True)
                # Evaluate Answer
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Evaluate"):

                        if not st.session_state.user_answer:
                            st.warning("Enter an answer.")
                        elif st.session_state.vector_store is None:
                            st.error("Upload notes first.")
                        else:
                            with st.spinner("Grading..."):

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

                with col_b:
                    # NEXT QUESTION
                    if st.button("Next Question"):
                        st.session_state.user_answer = ""
                        st.session_state.pyq_evaluation = None
                        st.session_state.related_chunks = ""
                        st.session_state.pyq_index += 1
                        st.rerun()

            if st.session_state.pyq_evaluation:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container():
                    st.markdown("#### Feedback")
                    st.write(st.session_state.pyq_evaluation)

        else:
            st.info("End of questions.")
            if st.button("Restart"):
                st.session_state.pyq_index = 0
                st.rerun()
