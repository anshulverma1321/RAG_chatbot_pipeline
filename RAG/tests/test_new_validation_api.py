import os
import sys
import json
import time

# Ensure root folder is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app

def main():
    client = TestClient(app)
    
    test_images = {
        "paragraph.png": "TEXT_IMAGE",
        "chart.png": "CHART",
        "arch.jpg": "DIAGRAM",
        "infographics.jpg": "MIXED",
        "scenery.jpg": "NATURAL_IMAGE"
    }

    images_dir = os.path.join(BASE_DIR, "test", "images")
    
    print("=" * 60)
    print("      NEW MULTIMODAL VALIDATION API TEST SUITE")
    print("=" * 60)

    # 1. Test POST /validation/image-classify
    print("\n--- Testing POST /validation/image-classify ---")
    for img_name, expected_type in test_images.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] File not found: {img_path}")
            continue

        ext = os.path.splitext(img_name)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        if ext == ".webp":
            mime = "image/webp"

        with open(img_path, "rb") as f:
            files = {"file": (img_name, f, mime)}
            response = client.post("/validation/image-classify", files=files)
        
        if response.status_code == 200:
            res_data = response.json()
            print(f"File: {img_name:<18} | Classified as: {res_data.get('image_type'):<13} | Expected: {expected_type:<13} | Confidence: {res_data.get('confidence'):.4f}")
        else:
            print(f"File: {img_name:<18} | Classification failed with status: {response.status_code}")

    # 2. Test POST /validation/image
    print("\n--- Testing POST /validation/image ---")
    for img_name, expected_type in test_images.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        ext = os.path.splitext(img_name)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        print(f"\nProcessing upload for {img_name}...")
        t0 = time.perf_counter()
        with open(img_path, "rb") as f:
            files = {"file": (img_name, f, mime)}
            response = client.post("/validation/image", files=files)
        elapsed = time.perf_counter() - t0

        if response.status_code == 200:
            res_data = response.json()
            print(f"  Classification : {res_data.get('image_type')} (Confidence: {res_data.get('confidence'):.2f})")
            print(f"  Extractor Used : {res_data.get('extractor_selected')}")
            print(f"  Knowledge Keys : {list(res_data.get('knowledge', {}).keys())}")
            print(f"  Timings        : {res_data.get('timings')}")
            print(f"  Request Time   : {elapsed:.2f}s")
            
            # Print a snippet of rich_text_representation
            rich_text = res_data.get("rich_text_representation", "")
            preview = rich_text.split('\n')[:8]
            print("  Rich Text Preview:")
            print("\n".join(["    " + line for line in preview]))
        else:
            print(f"  Failed: {response.status_code} - {response.text}")

    # 3. Test POST /validation/image-debug
    print("\n--- Testing POST /validation/image-debug ---")
    # Just test with one image
    debug_img = "chart.png"
    img_path = os.path.join(images_dir, debug_img)
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            files = {"file": (debug_img, f, "image/png")}
            response = client.post("/validation/image-debug", files=files)
        if response.status_code == 200:
            res_data = response.json()
            print(f"Debug classification keys : {list(res_data.get('classification', {}).keys())}")
            print(f"Selected route             : {res_data.get('selected_route')}")
            print(f"Raw extractor keys         : {list(res_data.get('raw_extractor_output', {}).keys())}")
            print(f"Debug timings              : {res_data.get('timings')}")
        else:
            print(f"Debug endpoint failed: {response.status_code}")

    # 4. Test POST /validation/pdf-images
    print("\n--- Testing POST /validation/pdf-images ---")
    pdf_path = os.path.join(BASE_DIR, "RAG", "data", "test_images.pdf")
    if os.path.exists(pdf_path):
        print(f"Processing PDF images for test_images.pdf...")
        t0 = time.perf_counter()
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            response = client.post("/validation/pdf-images", files=files)
        elapsed = time.perf_counter() - t0
        
        if response.status_code == 200:
            res_data = response.json()
            print(f"Total images found in PDF: {res_data.get('total_images')}")
            for img_item in res_data.get("images", []):
                print(f"  Page: {img_item.get('page')} | Index: {img_item.get('image_index')} | Type: {img_item.get('image_type')} | Extractor: {img_item.get('extractor')}")
            print(f"Total time for PDF extraction & process: {elapsed:.2f}s")
        else:
            print(f"PDF images endpoint failed: {response.status_code}")
    else:
        print("[WARN] test_images.pdf not found.")

if __name__ == "__main__":
    main()
