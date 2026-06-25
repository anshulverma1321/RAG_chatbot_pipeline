import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

from RAG.ingestion import describe_image_with_gemini

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
        print("Or provide a custom image path: python test_image_summarization.py <image_path>")
        sys.exit(1)
        
    print(f"Running image summarization on: {image_path}")
    
    # 2. Ensure output directory exists
    out_dir = os.path.join(base_dir, "outputs", "image_summaries")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Read image and run VLM summarization
    filename = os.path.basename(image_path)
    mime_type = "image/png"
    if filename.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
        
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            
        print("\nImage:")
        print(filename)
        
        print("\nGenerating Summary (calling Gemini)...")
        summary = describe_image_with_gemini(img_bytes, mime_type)
        
        print("\nGenerated Summary:")
        safe_print(summary)
        
        # Save output
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        out_file = os.path.join(out_dir, txt_filename)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(summary)
            
        print(f"\nSummary saved successfully to: {out_file}")
    except Exception as e:
        print(f"Error generating image summary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
