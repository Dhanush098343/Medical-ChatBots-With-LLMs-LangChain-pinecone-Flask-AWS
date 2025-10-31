from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore


load_dotenv(dotenv_path=".env", override=True)

pinecone_apikey=os.getenv("pinecone_apikey")
groq_apikey=os.getenv("groq_apikey")

os.environ["PINECONE_API_KEY"] = pinecone_apikey
os.environ["GROQ_API_KEY"] = groq_apikey

extracted_data=load_pdf_files("data/")
minimal_docs= filter_to_minimal_docs(extracted_data)
texts_chunk=text_split(minimal_docs)

embedding= download_embeddings()
pc=Pinecone(api_key=pinecone_apikey)

index_name="medical-chatbot-index"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec= ServerlessSpec(cloud='aws',region='us-east-1')
    )
index= pc.Index(index_name)


docsearch= PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
    )