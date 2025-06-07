import streamlit as st
import re
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load secrets
USERNAME = st.secrets["AUTH_USERNAME"]
PASSWORD = st.secrets["AUTH_PASSWORD"]

def check_auth():
    def do_login():
        if (st.session_state.get("username") == USERNAME and
                st.session_state.get("password") == PASSWORD):
            st.session_state["auth_ok"] = True
        else:
            st.error("Invalid credentials")

    if not st.session_state.get("auth_ok"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=do_login)
        st.stop()

check_auth()

PRIMARY_BLUE = "#003366"
ACCENT_BLUE = "#0072CE"
BG_COLOR = "#f5f9fc"
TEXT_COLOR = "#0a1f44"

st.set_page_config(page_title="Brook Aviation Demo", page_icon="🚁️", layout="wide")

st.markdown(
    f"""
    <style>
    body {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {PRIMARY_BLUE};
        border-radius: 6px;
        padding: 8px;
        font-size: 1rem;
    }}
    .stButton>button {{
        background-color: {PRIMARY_BLUE};
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 20px;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {ACCENT_BLUE};
        cursor: pointer;
    }}
    .source-chunk {{
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-left: 6px solid {PRIMARY_BLUE};
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        white-space: pre-wrap;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        color: {TEXT_COLOR};
    }}
    mark {{
        background-color: {ACCENT_BLUE};
        color: white;
        padding: 0 4px;
        border-radius: 3px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚁 Brook Aviation Demo")

uploaded_files = st.file_uploader("Upload one or more PDF files for analysis", type="pdf", accept_multiple_files=True)

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def highlight_query(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

if uploaded_files and len(uploaded_files) >= 1:
    all_docs = []
    for file in uploaded_files:
        temp_file_path = save_uploaded_file(file)
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        for doc in docs:
            doc.metadata["source_file"] = file.name
        all_docs.extend(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    query = st.text_input("Enter your question about the documents")

    if query:
        with st.spinner("Processing your question..."):
            result = qa_chain(query)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Answer")
            st.write(result["result"])

        with col2:
            st.subheader("Top Source Passages")
            for doc in result["source_documents"][:3]:
                content = doc.page_content.strip().replace("\n", " ")
                content_highlighted = highlight_query(content, query)
                source_file = doc.metadata.get("source_file", "Unknown file")
                page_num = doc.metadata.get("page", "Unknown page")

                st.markdown(
                    f'<div class="source-chunk"><b>{source_file} - Page {page_num}</b><br>{content_highlighted[:800]}...</div>',
                    unsafe_allow_html=True,
                )
else:
    st.info("Please upload at least one PDF file to start.")
