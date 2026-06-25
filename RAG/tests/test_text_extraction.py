import os
import sys
import pdfplumber

# Ensure root folder is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    # Find project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        print("Please provide a PDF file path: python test_text_extraction.py <pdf_path>")
        sys.exit(1)
        
    print(f"Running text extraction on: {pdf_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "text")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Extract text
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = pdf.pages[page_idx]
                text = page.extract_text() or ""
                
                # Print output formatting safely
                safe_print(f"\nPage {page_num}")
                safe_print(f"Characters: {len(text)}")
                
                preview = text[:200].replace('\n', ' ') + ("..." if len(text) > 200 else "")
                safe_print(f"Preview:\n{preview}")
                
                # Save output
                out_file = os.path.join(out_dir, f"page_{page_num}.txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
        print(f"\nText extraction complete. Output files saved in: {out_dir}")
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
