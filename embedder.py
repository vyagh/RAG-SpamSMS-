from sentence_transformers import SentenceTransformer
from config import MODELS

def embed_chunks(chunks, model_name=MODELS['embedding']):
    model = SentenceTransformer(model_name)
    texts = [chunk[2] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=MODELS['batch_size'], show_progress_bar=True)
    return list(zip([chunk[1] for chunk in chunks], embeddings))

# TESTING CODE
'''if __name__ == "__main__":
    from data_ingest import ingest_data
    from custom_chunker import chunk_documents
    embeddings = embed_chunks(chunk_documents(ingest_data()))
    print(f"Total embeddings: {len(embeddings)}")
    print("Sample embedding shape:", embeddings[0][1].shape)'''