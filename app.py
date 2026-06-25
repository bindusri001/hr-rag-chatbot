import streamlit as st
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["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"]
os.environ["LANGCHAIN_PROJECT"] = "hr-rag-chatbot"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# GROQ API KEY
os.environ["GROQ_API_KEY"] = "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"

st.title("Zyro Dynamics HR Help Desk")

# Load PDFs
documents = []

pdf_files = [file for file in os.listdir(".") if file.endswith(".pdf")]

for file in pdf_files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

# Check documents
if len(documents) == 0:
    st.error("No PDF documents loaded.")
    st.stop()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Check chunks
if len(chunks) == 0:
    st.error("No text chunks created.")
    st.stop()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# Chat Function
def ask_hr_bot(question):

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are Zyro Dynamics HR Help Desk Assistant.

Answer ONLY from the provided HR policy documents.

If the answer is not found, say:
"I could not find this information in the HR policies."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content

question = st.text_input("Ask an HR Question")

if question:
    answer = ask_hr_bot(question)
    st.write(answer)
