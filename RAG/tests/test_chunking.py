import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

import pdfplumber
from RAG.ingestion import split_text

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    data_dir = os.path.join(base_dir, "RAG", "data")
    
    # 1. Determine input PDF
    pdf_path = None
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default to first PDF found in data_dir
        if os.path.exists(data_dir):
            pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
            if pdfs:
                pdf_path = os.path.join(data_dir, pdfs[0])
                
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"Error: No input PDF file found at: {pdf_path}")
        sys.exit(1)
        
    print(f"Running chunking analysis on: {pdf_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "chunks")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Extract text and split
    chunk_size = 800
    overlap = 150
    
    try:
        all_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap)
                all_chunks.extend(chunks)
                
        print(f"\nChunk Size: {chunk_size}")
        print(f"Overlap: {overlap}")
        print(f"\nTotal Chunks: {len(all_chunks)}")
        
        for idx, chunk in enumerate(all_chunks, 1):
            safe_print(f"\nChunk {idx}:")
            safe_print(chunk)
            safe_print("-" * 30)
            
            # Save chunk
            out_file = os.path.join(out_dir, f"chunk_{idx}.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(chunk)
                
        print(f"\nChunking complete. Output files saved in: {out_dir}")
    except Exception as e:
        print(f"Error chunking document: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
