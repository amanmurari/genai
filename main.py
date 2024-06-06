import os
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

key="aman"
os.environ["GROQ_API_KEY"]=key
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
model="Llama3-8b-8192"
add_routes(
    app,
    ChatGroq(model=model),
    path="/openai",
)

add_routes(
    app,
  ChatGroq(model=model),
    path="/an",
)

model = ChatGroq(model=model)
prompt = ChatPromptTemplate.from_template("""Your task is to generate a short summary of a given topic .
 
Summarize the topic below, delimited by triple 
quortation '''{topic}'''""")
add_routes(
    app,
    prompt | model,
    path="/summerize",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8800)
