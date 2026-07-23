"""
Builds a local SQLite database for the MedQuAD dataset using FTS5 for fast full-text search.
This allows Sanjeevani to search 47k+ genuine NIH medical Q&As offline instantly.
"""

import sqlite3
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets first: pip install datasets")
    exit(1)

DB_PATH = Path(__file__).parent / "medquad.sqlite"

def build_db():
    print(f"Downloading MedQuAD dataset from Hugging Face...")
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    
    print(f"Loaded {len(dataset)} Q&A pairs. Building SQLite database at {DB_PATH}...")
    
    # Remove existing DB if it exists
    if DB_PATH.exists():
        DB_PATH.unlink()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create an FTS5 virtual table for fast full-text searching
    cursor.execute("""
        CREATE VIRTUAL TABLE medquad USING fts5(
            qtype, 
            question, 
            answer
        )
    """)
    
    # Insert data
    start_time = time.time()
    rows = []
    for item in dataset:
        rows.append((item.get("qtype", ""), item.get("Question", ""), item.get("Answer", "")))
        
    cursor.executemany("INSERT INTO medquad (qtype, question, answer) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    
    print(f"Successfully built MedQuAD database in {time.time() - start_time:.2f} seconds.")
    print(f"Database size: {DB_PATH.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    build_db()
