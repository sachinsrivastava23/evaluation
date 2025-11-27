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
#   UI CONFIGURATION & CSS (THE NUCLEAR FIX)
# ------------------------------------------------------

st.set_page_config(page_title="Zen Study Companion", page_icon="🧘", layout="centered")

def inject_custom_css():
    st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Merriweather:wght@300;700&display=swap');

        /* --- 1. THE NUCLEAR TEXT FIX --- */
        /* This forces ALL text in the app to be dark gray, overriding Dark Mode white text */
        
        .stApp, .stApp > header {
            background-color: #dfe6e9 !important; /* Force background color */
        }

        /* Target every possible text element to force it Dark */
        h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {
            color: #2d3436 !important;
        }

        /* --- 2. EXCEPTIONS (Revert text to white for specific dark buttons) --- */
        
        /* The "Browse Files" button inside File Uploader */
        button[data-testid="baseButton-secondary"] {
            color: #2d3436 !important; /* Keep dark text on light button */
            border-color: #2d3436 !important;
        }

        /* The Main Action Buttons (Purple) - Text must be white */
        .stButton > button {
            background-color: #6c5ce7 !important;
            border: none !important;
        }
        .stButton > button p {
            color: #ffffff !important; /* Force text white on purple button */
        }
        
        /* --- 3. COMPONENT VISIBILITY FIXES --- */
        
        /* Fix the Radio Buttons (The bubbles themselves) */
        div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            color: #2d3436 !important;
        }

        /* Fix the File Uploader "Drag and drop" text */
        section[data-testid="stFileUploaderDropzone"] div, 
        section[data-testid="stFileUploaderDropzone"] span, 
        section[data-testid="stFileUploaderDropzone"] small {
            color: #2d3436 !important;
        }

        /* --- 4. CARD STYLING --- */
        .css-card {
            background-color: #FFFFFF !important;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            border-top: 6px solid #6c5ce7;
        }

        /* --- 5. INPUT FIELDS --- */
        /* Force input backgrounds to be light so dark text is visible */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #f1f2f6 !important;
            color: #2d3436 !important;
            border: 1px solid #ced6e0 !important;
        }
        
        /* --- 6. COLORED BOXES --- */
        .question-box {
            background-color: #fff200 !important;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            border-left: 5px solid #ffa502;
        }
        .question-box div, .question-box p {
             color: #2d3436 !important;
        }

        .eval-box {
            background-color: #55efc4 !important;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1.5rem;
            border-left: 5px solid #00b894;
        }
        .eval-box div, .eval-box h3 {
             color: #006266 !important;
        }

        /* HIDE DEFAULT HEADER */
        header {visibility: hidden;}
        footer {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

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
#                   INITIAL SETUP
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
#             GOOGLE SEARCH API FUNCTION
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
#               DOCUMENT CHUNKING
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
#          PDF TEXT AUTO-DETECTOR (NO OCR)
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

# Title Area
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color:#2d3436 !important;">🧘 Zen Study Companion</h1>
    <p style="color: #636e72 !important; font-size: 1.1rem;">Focus. Learn. Repeat.</p>
</div>
""", unsafe_allow_html=True)

# Container for Upload
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 📂 Step 1: Upload Material")
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing your study material... 🍃"):
        docs = load_pdf_auto(tmp_path)
        chunks = chunk_documents(docs)
        chunk_docs = to_document(chunks)

        # create simple RAM vector store
        vs = SimpleVectorStore(embeddings)
        vs.add_documents(chunk_docs)

        st.session_state.vector_store = vs
        st.session_state.chunks = chunks

    st.success("✨ Material ready! Embeddings stored in memory.")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
#                QUESTION GENERATION UI
# ------------------------------------------------------

st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 🎯 Step 2: Choose Mode")

question_mode = st.radio(
    "Select How You Want To Get Questions",
    ("Generated From the Notes", "PYQ"),
    label_visibility="collapsed",
    horizontal=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
#            MODE 1: GENERATED FROM NOTES
# ------------------------------------------------------

if question_mode == "Generated From the Notes":
    
    # Logic variables (unchanged)
    if "generated_question" not in st.session_state:
        st.session_state.generated_question = None

    if "current_chunk" not in st.session_state:
        st.session_state.current_chunk = None

    if "user_ans" not in st.session_state:
        st.session_state.user_ans = ""

    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    # UI Wrapper
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("✨ Generate New Question"):
            if not st.session_state.chunks:
                st.warning("⚠️ Please upload a PDF first.")
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

                with st.spinner("Thinking of a question..."):
                    output = model.invoke(prompt)
                    st.session_state.generated_question = output.content

                st.session_state.user_ans = ""
                st.session_state.evaluation = None

    if st.session_state.generated_question:
        st.markdown(f"""
        <div class="question-box">
            <div style="font-size:0.9rem; color:#d35400 !important; margin-bottom:5px;">QUESTION</div>
            <div style="color:#2d3436 !important;">{st.session_state.generated_question}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Your Answer:**")
        st.session_state.user_ans = st.text_area(
            "TYPE THE ANSWER...",
            value=st.session_state.user_ans,
            height=150,
            label_visibility="collapsed",
            placeholder="Type your understanding here..."
        )

        if st.button("📝 Submit Answer"):
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
            
            with st.spinner("Grading..."):
                evaluation = model.invoke(evaluation_prompt)
                st.session_state.evaluation = evaluation.content

    if st.session_state.evaluation:
        st.markdown(f"""
        <div class="eval-box">
            <h3 style="color:#006266 !important;">📊 Evaluation Results</h3>
            <div style="color:#006266 !important;">{st.session_state.evaluation.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
#                    MODE 2: PYQ (ABESIT)
# ------------------------------------------------------

if question_mode == "PYQ":
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### 🏦 Question Bank Search")

    # State Setup (unchanged)
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

    # Inputs
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

    # Search Button
    if st.button("🔍 Find Questions"):

        if not subject:
            st.error("Please enter a Subject Name or Code.")
        else:
            with st.spinner(f"Scanning ABESIT Library for {course} {subject}..."):

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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Display Questions
    if st.session_state.pyq_list:
        
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        idx = st.session_state.pyq_index
        total = len(st.session_state.pyq_list)

        if idx < total:
            st.markdown(f"#### Question {idx + 1} of {total}")
            
            st.session_state.current_question = st.session_state.pyq_list[idx]
            
            st.markdown(f"""
            <div class="question-box">
                <div style="font-size:0.9rem; color:#d35400 !important; margin-bottom:5px;">QUESTION</div>
                <div style="color:#2d3436 !important;">{st.session_state.current_question}</div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.user_answer = st.text_area(
                "Your Answer:",
                value=st.session_state.user_answer,
                height=150,
                placeholder="Write your answer here..."
            )

            col_a, col_b = st.columns(2)
            
            # Evaluate Answer
            with col_a:
                if st.button("📝 Evaluate Answer"):

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

            # NEXT QUESTION
            with col_b:
                if st.button("➡️ Next Question"):
                    st.session_state.user_answer = ""
                    st.session_state.pyq_evaluation = None
                    st.session_state.related_chunks = ""
                    st.session_state.pyq_index += 1
                    st.rerun()
            
            if st.session_state.pyq_evaluation:
                st.markdown(f"""
                <div class="eval-box">
                    <h3 style="color:#006266 !important;">📊 Evaluation Results</h3>
                    <div style="color:#006266 !important;">{st.session_state.pyq_evaluation.replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding: 2rem;">
                <h3>🎉 Excellent work!</h3>
                <p>You've completed all the questions in this set.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Start Over"):
                st.session_state.pyq_index = 0
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
