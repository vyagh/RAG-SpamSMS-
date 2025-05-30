import pandas as pd
import re
from config import SPAM_CSV_PATH

def ingest_data(file_path=SPAM_CSV_PATH):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df['text'] = df['text'].apply(lambda t: re.sub(r'<.*?>|\s+', ' ', t).strip())
    df['id'] = df.index
    df['metadata'] = df.apply(lambda r: {'label': r['label']}, axis=1)
    return df[['id', 'text', 'metadata']].dropna()

# TESTING CODE
'''if __name__ == "__main__":
    df = ingest_data()
    print(f"DataFrame shape: {df.shape}")
    print("\nSample row:", df.iloc[0].to_dict())'''