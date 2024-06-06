import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import PyPDFLoader
from starlette.types import Receive

os.environ["LANGCHAIN_API_KEY"]="aman"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="simple"
os.environ["GROQ_API_KEY"]="aman"
os.environ["OPENAI_API_KEY"]="aman"
client= ChatGroq()
loader=PyPDFLoader("./aman.pdf").load()
page=RecursiveCharacterTextSplitter(chunk_size=500).split_documents(loader)
embedding= HuggingFaceEmbeddings()
base= FAISS.from_documents(page,embedding)

base.retrive
