import os
import sys
import json
import time
import argparse

# Resolve project root and make it importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from RAG.services.text_knowledge_extractor import extract_text_knowledge

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))

def get_mime_type(image_path: str) -> str:
    """Derives the MIME type from the file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png":  "image/png",
    }
    return mime_map.get(ext, "image/png")

def main():
    parser = argparse.ArgumentParser(
        description="Text Image Pipeline Validation (OCR and Cleaning only)"
    )
    parser.add_argument(
        "image_path",
        help="Path to the text image file to process"
    )
    args = parser.parse_args()
    
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)
        
    mime_type = get_mime_type(image_path)
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        
    t_start = time.perf_counter()
    result = extract_text_knowledge(image_bytes, mime_type)
    t_total = time.perf_counter() - t_start
    
    output_data = {
        "ocr_engine_used": result.get("ocr_engine_used"),
        "average_confidence": result.get("average_confidence"),
        "ocr_raw_text": result.get("ocr_raw_text"),
        "cleaned_text": result.get("cleaned_text"),
        "word_count": result.get("word_count"),
        "timing_breakdown": {
            "ocr_execution_time_seconds": result.get("ocr_execution_time"),
            "gemini_cleaning_time_seconds": result.get("gemini_cleaning_execution_time"),
            "total_time_seconds": round(t_total, 4)
        }
    }
    
    # Pretty-print JSON output
    safe_print(json.dumps(output_data, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
