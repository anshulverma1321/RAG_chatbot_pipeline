import os
import sys

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
    images_dir = os.path.join(base_dir, "outputs", "images")
    
    # 1. Determine input image
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to first image found in images_dir
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if images:
                image_path = os.path.join(images_dir, images[0])
                
    if not image_path or not os.path.exists(image_path):
        print(f"Error: No image found. Please run test_image_extraction.py first to extract images to: {images_dir}")
        print("Or provide a custom image path: python test_ocr.py <image_path>")
        sys.exit(1)
        
    print(f"Running OCR validation on: {image_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "ocr")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Detect and run OCR engines
    ocr_text = ""
    engine_used = None
    
    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        print("Initializing PaddleOCR (primary)...")
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_path, cls=True)
        
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                text_lines.append(line[1][0])
        ocr_text = "\n".join(text_lines)
        engine_used = "PaddleOCR"
    except (ImportError, Exception) as paddle_err:
        print(f"PaddleOCR is not available. (Error: {paddle_err})")
        print("Attempting EasyOCR fallback...")
        try:
            import easyocr
            print("Initializing EasyOCR...")
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_path)
            text_lines = [line[1] for line in result]
            ocr_text = "\n".join(text_lines)
            engine_used = "EasyOCR"
        except (ImportError, Exception) as easy_err:
            print("\n[Warning] Neither PaddleOCR nor EasyOCR is installed or functioning in this environment.")
            print("To run this OCR validation test, please install one of them:")
            print("Option A (PaddleOCR): pip install paddlepaddle paddleocr")
            print("Option B (EasyOCR): pip install easyocr")
            sys.exit(1)
            
    print("\nImage:")
    print(os.path.basename(image_path))
    print(f"\nOCR Text (via {engine_used}):")
    safe_print(ocr_text if ocr_text else "[No text detected]")
    
    # Save results
    filename = os.path.basename(image_path)
    txt_filename = os.path.splitext(filename)[0] + "_ocr.txt"
    out_file = os.path.join(out_dir, txt_filename)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(ocr_text)
        
    print(f"\nOCR extraction saved to: {out_file}")

if __name__ == "__main__":
    main()
