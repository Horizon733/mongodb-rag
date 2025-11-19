import streamlit as st
from pymongo import MongoClient
import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo.operations import SearchIndexModel

# MongoDB Atlas connection
MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
client = MongoClient(MONGO_URI)
db = client["ragdb"]
collection = db["documents"]

# Embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if vector index exists, create if not
existing_indexes = [idx["name"] for idx in collection.list_search_indexes()]
if "vector_index" not in existing_indexes:
    search_index_model = SearchIndexModel(
      definition = {
        "fields": [
          {
            "type": "vector",
            "numDimensions": 384,
            "path": "embedding",
            "similarity": "cosine"
          }
        ]
      },
      name = "vector_index",
      type = "vectorSearch"
    )
    collection.create_search_index(model=search_index_model)

# Ollama LLM endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

def embed_text(text):
    return embedder.embed_query(text)

def similarity_search(query_embedding, top_k=3):
    # MongoDB Atlas vector search
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": top_k,
                "index": "vector_index"
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return [doc["text"] for doc in results]

def ollama_generate(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    if response.ok:
        return response.json().get("response", "")
    return "LLM error."

def process_pdf_and_store(pdf_path):
    # Load and chunk PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        text = chunk.page_content
        embedding = embed_text(text)
        collection.insert_one({
            "text": text,
            "embedding": embedding
        })

st.title("MongoDB Atlas RAG with Ollama LLM")

user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Searching..."):
        query_emb = embed_text(user_query)
        docs = similarity_search(query_emb)
        context = "\n".join(docs)
        print("Retrieved Context:", context)
        answer = ollama_generate(context, user_query)
    # st.markdown("### Retrieved Context")
    # st.write(context)
    st.markdown("### LLM Answer")
    st.write(answer)

st.sidebar.header("PDF Upload & Index")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    with st.spinner("Processing PDF and storing chunks..."):
        process_pdf_and_store("temp.pdf")
    st.sidebar.success("PDF processed and stored in MongoDB.")
    