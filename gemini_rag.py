import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

key = "aman"
GOOGLE_API_KEY = "aman"
client = ChatGroq(groq_api_key=key, model="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

""")


# vector embeddings
def vector_embd():
    if "vector" not in st.session_state:
        st.session_state.direct = PyPDFDirectoryLoader("./pdfdir")
        st.session_state.loader = st.session_state.direct.load()
        st.session_state.tspliter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200)
        st.session_state.split = st.session_state.tspliter.split_documents(
            st.session_state.loader)
        st.session_state.emned = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, model="models/embeddings-001")
        st.session_state.vector = FAISS.from_documents(st.session_state.split,
                                                       st.session_state.emned)


prompt1 = st.text_input("Enter Your Question From Doduments")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(client, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
