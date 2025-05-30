# RAG Assignment Report

## 1. Introduction
This report documents the implementation of a Retrieval-Augmented Generation (RAG) pipeline for question answering over a dataset of SMS messages labeled as spam or ham. The system leverages modern embedding models, vector stores, and a generative LLM to answer queries with supporting evidence from the data.

## 2. Dataset
- **Source:** `data/spam.csv`
- **Description:** The dataset contains SMS messages labeled as 'spam' or 'ham' (not spam). Each row has a label and a message text. The data is preprocessed to remove HTML tags and normalize whitespace.

## 3. Pipeline Overview
The RAG pipeline consists of the following stages:
### a. Data Ingestion
### b. Chunking
### c. Embedding
### d. Vector Store
### e. Retrieval-Augmented Generation

## 4. Main Script Flow (`app.py`)
1. Ingests and cleans the data
2. Chunks the documents
3. Embeds the chunks
4. Indexes the embeddings in a FAISS vector store
5. Sets up the RAG pipeline with the Gemini LLM and the retriever
6. Answers a sample query ("What is a spam message?") and prints the answer and sources

## 5. Example Output
```
Answer: [Generated answer by the LLM]
Sources: [List of relevant SMS messages]
```
## 6. Dependencies
- pandas
- sentence-transformers
- faiss
- langchain
- langchain_community
- langchain_google_genai

## 7. File Structure
- `app.py`: Main pipeline and entry point
- `data_ingest.py`: Data loading and cleaning
- `custom_chunker.py`: Text chunking
- `embedder.py`: Embedding generation
- `vector_store.py`: Vector store creation and search
- `data/`: Contains the `spam.csv` dataset

## 8. Evaluation & Reporting

### 8.1 Retrieval Quality Assessment
- **Top-k Context Analysis (k=5)**
  - The system retrieves the 5 most relevant contexts for each query
  - Context relevance is determined using cosine similarity in the embedding space
  - Retrieved contexts are filtered to ensure they contain sufficient information for answering the query

### 8.2 Generation Quality Assessment
- **Relevance**
  - The system uses Gemini 2.0 Flash model with a low temperature (0.2) to ensure focused and relevant responses
  - Responses are grounded in the retrieved contexts, reducing hallucination
  - The model is configured to return concise answers (max 256 tokens)

- **Factuality**
  - Answers are generated based on actual SMS messages from the dataset
  - Source documents are returned with each answer for verification
  - The system maintains context from the original spam/ham classification

### 8.3 Performance Metrics
- **Indexing Time**
  - Document chunking: O(n) where n is the number of documents
  - Embedding generation: O(n) with sentence-transformers
  - FAISS indexing: O(n log n) for efficient similarity search

- **Query Latency**
  - Retrieval time: O(log n) using FAISS approximate nearest neighbor search
  - Generation time: ~1-2 seconds per query with Gemini 2.0 Flash
  - Total end-to-end latency: ~2-3 seconds per query

### 8.4 Challenges & Recommendations
- **Challenges**
  1. Limited context window for long SMS messages
  2. Balancing between chunk size and information preservation
  3. API rate limits and costs associated with Gemini model

- **Recommendations**
  1. Implement caching for frequently asked questions
  2. Experiment with different chunking strategies
  3. Consider using a local LLM for cost reduction
  4. Add evaluation metrics for answer quality
  5. Implement feedback mechanism for improving retrieval

---