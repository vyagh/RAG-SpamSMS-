from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths & API
DATA_DIR = Path('./data')
SPAM_CSV_PATH = DATA_DIR / 'spam.csv'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Models & Parameters
MODELS = {
    'embedding': 'all-MiniLM-L6-v2',
    'llm': "gemini-2.0-flash",
    'temperature': 0.2,
    'max_tokens': 256,
    'chunk_size': 500,
    'top_k': 5,
    'batch_size': 32
}