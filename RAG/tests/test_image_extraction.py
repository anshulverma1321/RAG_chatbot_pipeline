import os
import sys

# Ensure root folder is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pypdf import PdfReader

def main():
    # Find project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "RAG", "data")
    
    # 1. Determine input PDF
    pdf_path = os.path.join(data_dir, "documentation.pdf")
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
        
    print(f"Running image extraction on: {pdf_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "images")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Scan and extract images
    images_found_total = 0
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # First pass: Count total images
        for page in reader.pages:
            if hasattr(page, "images"):
                images_found_total += len(page.images)
                
        print(f"\nImages Found: {images_found_total}")
        
        image_idx = 1
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = reader.pages[page_idx]
            
            if hasattr(page, "images"):
                for img_in_page_idx, img in enumerate(page.images):
                    # Determine extension
                    ext = ".png"
                    if img.name.lower().endswith(".jpg") or img.name.lower().endswith(".jpeg"):
                        ext = ".jpg"
                        
                    filename = f"page_{page_num}_image_{image_idx}{ext}"
                    out_file = os.path.join(out_dir, filename)
                    
                    with open(out_file, "wb") as f:
                        f.write(img.data)
                        
                    print(f"Page {page_num}")
                    print(f"Image {image_idx}")
                    print("Saved Successfully")
                    print("-" * 15)
                    
                    image_idx += 1
                    
        print(f"\nImage extraction complete. Output files saved in: {out_dir}")
    except Exception as e:
        print(f"Error extracting images: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
