import os
import sys
import csv

# Ensure root folder is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pdfplumber
from RAG.ingestion import table_to_markdown

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def save_table_as_csv(table, file_path):
    """Saves a raw pdfplumber table list to a CSV file."""
    if not table:
        return
    cleaned = [[str(cell or "").strip() for cell in row] for row in table]
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(cleaned)
    except Exception as e:
        print(f"Error saving CSV table: {e}")

def main():
    # Find project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "RAG", "data")
    
    # 1. Determine input PDF
    pdf_path = os.path.join(data_dir, "test2.pdf")
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
    if not os.path.exists(pdf_path):
        # Fallback to any PDF in data
        pdf_path = None
        if os.path.exists(data_dir):
            pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
            if pdfs:
                pdf_path = os.path.join(data_dir, pdfs[0])
                
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"Error: Input PDF file not found at: {pdf_path}")
        sys.exit(1)
        
    print(f"Running table extraction on: {pdf_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Extract tables
    tables_found_total = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # First pass: Count total tables across all pages
            for page in pdf.pages:
                tables = page.extract_tables()
                tables_found_total += len(tables)
                
            print(f"\nTables Found: {tables_found_total}")
            
            table_idx = 1
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = pdf.pages[page_idx]
                tables = page.extract_tables()
                
                for t in tables:
                    md_table = table_to_markdown(t)
                    if md_table:
                        safe_print(f"\nTable {table_idx} (Page {page_num}):")
                        safe_print(md_table)
                        safe_print("-" * 40)
                        
                        # Save markdown output
                        md_file = os.path.join(out_dir, f"page_{page_num}_table_{table_idx}.md")
                        with open(md_file, "w", encoding="utf-8") as f:
                            f.write(md_table)
                            
                        # Save CSV output
                        csv_file = os.path.join(out_dir, f"page_{page_num}_table_{table_idx}.csv")
                        save_table_as_csv(t, csv_file)
                        
                        table_idx += 1
                        
        print(f"\nTable extraction complete. Output files saved in: {out_dir}")
    except Exception as e:
        print(f"Error extracting tables: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
