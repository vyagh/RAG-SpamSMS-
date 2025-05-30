from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import MODELS
import os

def index_embeddings(chunks, embeddings, embedding_model=MODELS['embedding']):
    texts = [chunk[2] for chunk in chunks]
    metadatas = [chunk[3] for chunk in chunks]
    return FAISS.from_texts(texts, HuggingFaceEmbeddings(model_name=embedding_model), metadatas=metadatas)

def save_vectorstore(vectorstore, path="./data/vectorstore"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)
    return path

def load_vectorstore(path="./data/vectorstore", embedding_model=MODELS['embedding']):
    return FAISS.load_local(path, HuggingFaceEmbeddings(model_name=embedding_model), allow_dangerous_deserialization=True)

# TESTING CODE
if __name__ == "__main__":
    from data_ingest import ingest_data
    from custom_chunker import chunk_documents
    from embedder import embed_chunks
    
    chunks = chunk_documents(ingest_data())
    vectorstore = index_embeddings(chunks, embed_chunks(chunks))
    save_path = save_vectorstore(vectorstore)
    
    results = vectorstore.similarity_search("spam message example", k=1)
    print(f"Vector store saved to: {save_path}")
    print("Sample search result:", results[0].page_content[:100])