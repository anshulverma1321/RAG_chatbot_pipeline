import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

from RAG.services.image_intelligence import run_visual_understanding_logic

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
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
        print("Or provide a custom image path: python test_visual_understanding.py <image_path>")
        sys.exit(1)
        
    print(f"Running combined visual understanding on: {image_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "visual_understanding")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Read image and run pipelines
    filename = os.path.basename(image_path)
    ext = os.path.splitext(filename)[1].lower()
        
    try:
        print("\nExecuting visual understanding pipeline from services...")
        result = run_visual_understanding_logic(image_path, ext)
        ocr_text = result["ocr_text"]
        visual_summary = result["vision_summary"]
        combined_result = result["combined_understanding"]
        
        # Output
        print("\n" + "="*50)
        print("OCR Text:")
        safe_print(ocr_text if ocr_text else "[None]")
        print("\n" + "="*50)
        print("Visual Summary:")
        safe_print(visual_summary if visual_summary else "[None]")
        print("\n" + "="*50)
        print("Combined Result:")
        safe_print(combined_result)
        print("="*50)
        
        # Save output
        out_content = (
            f"OCR Text:\n{ocr_text}\n\n"
            f"Visual Summary:\n{visual_summary}\n\n"
            f"Combined Result:\n{combined_result}\n"
        )
        txt_filename = os.path.splitext(filename)[0] + "_merged.txt"
        out_file = os.path.join(out_dir, txt_filename)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(out_content)
            
        print(f"\nVisual understanding synthesis saved to: {out_file}")
    except Exception as e:
        print(f"Error during combined visual understanding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
