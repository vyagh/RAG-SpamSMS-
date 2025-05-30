from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import MODELS

def chunk_documents(df):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MODELS['chunk_size'],
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = []
    for _, row in df.iterrows():
        text_chunks = text_splitter.split_text(row['text'])
        for i, chunk in enumerate(text_chunks):
            chunks.append((
                row['id'],
                f"{row['id']}_{i}",
                chunk,
                row['metadata']
            ))
    return chunks

# TESTING CODE
'''if __name__ == "__main__":
    from data_ingest import ingest_data
    chunks = chunk_documents(ingest_data())
    print(f"Total chunks: {len(chunks)}")
    print("Sample chunk:", chunks[0])'''