from flask import Flask, render_template, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.helper import download_embeddings
from src.prompt import *
import os


app=Flask(__name__)
load_dotenv(dotenv_path=".env", override=True)

pinecone_apikey=os.getenv("pinecone_apikey")
groq_apikey=os.getenv("groq_apikey")

os.environ["PINECONE_API_KEY"] = pinecone_apikey
os.environ["GROQ_API_KEY"] = groq_apikey


embedding= download_embeddings()
index_name= "medical-chatbot-index"
docsearch= PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
    )

retriever= docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

prompt= ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)