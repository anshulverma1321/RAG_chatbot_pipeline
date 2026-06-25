import os
import sys
import json
import time
import re
import sqlite3
import logging

# Resolve project root and make it importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Custom log capturing handler
class IngestionLogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))

def main():
    db_path = os.path.join(BASE_DIR, "RAG", "data", "rag_tool.db")
    vector_db_path = os.path.join(BASE_DIR, "RAG", "data", "qdrant")
    pdf_path = os.path.join(BASE_DIR, "RAG", "data", "test_images.pdf")
    report_path = os.path.join(BASE_DIR, "outputs", "image_ingestion_report.json")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    print("=" * 60)
    print("      IMAGE INGESTION PIPELINE VALIDATION TEST")
    print("=" * 60)
    print(f"Database Path  : {db_path}")
    print(f"Vector DB Path : {vector_db_path}")
    print(f"PDF Path       : {pdf_path}")
    print(f"Report Path    : {report_path}\n")

    if not os.path.exists(pdf_path):
        print(f"[ERROR] Test PDF file not found at: {pdf_path}")
        print("Please run extraction or create test_images.pdf first.")
        sys.exit(1)

    # 1. Clean up existing document records to ensure clean ingestion
    filename = os.path.basename(pdf_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
    except Exception as db_init_err:
        print(f"[ERROR] Failed to query documents: {db_init_err}")
        row = None
    conn.close()

    if row:
        doc_id = row[0]
        print(f"Found existing document ID {doc_id} for '{filename}'. Deleting to run fresh ingestion...")
        from RAG.db import delete_document
        from RAG.vector_store import delete_vectors_by_doc
        try:
            delete_document(db_path, doc_id)
            delete_vectors_by_doc(vector_db_path, doc_id)
            print("Successfully deleted existing metadata and vector records.")
        except Exception as delete_err:
            print(f"[WARN] Failed to delete existing document: {delete_err}")
    else:
        print(f"No existing records found for '{filename}'. Starting clean.")

    # 2. Set up capturing log handler on RAG.ingestion logger
    ingest_logger = logging.getLogger("RAG.ingestion")
    ingest_logger.setLevel(logging.INFO)
    log_capture = IngestionLogCaptureHandler()
    ingest_logger.addHandler(log_capture)

    # Also capture logs from services to extract class results if needed
    service_logger = logging.getLogger("RAG.services")
    service_logger.setLevel(logging.INFO)
    service_logger.addHandler(log_capture)

    # 3. Execute the full PDF ingestion
    from RAG.ingestion import process_pdf
    
    t_start = time.perf_counter()
    try:
        new_doc_id, msg = process_pdf(pdf_path, db_path, vector_db_path)
        ingestion_success = True
        error_occurred = None
    except Exception as e:
        new_doc_id = None
        msg = str(e)
        ingestion_success = False
        error_occurred = str(e)
        print(f"[ERROR] Ingestion failed: {e}")

    t_elapsed = time.perf_counter() - t_start

    # Remove the log capture handlers
    ingest_logger.removeHandler(log_capture)
    service_logger.removeHandler(log_capture)

    # 4. Parse captured logs to find image classification and extraction details
    # Regex patterns to match log lines
    class_pattern = re.compile(
        r"Image classification success \| page=(\d+) \| idx=(\d+) \| type=(\w+) \| confidence=([\d.]+) \| reason=(.*)",
        re.IGNORECASE
    )
    class_fail_pattern = re.compile(
        r"Image classification failed \| page=(\d+) \| idx=(\d+) \| error=(.*)",
        re.IGNORECASE
    )
    extract_pattern = re.compile(
        r"Image extraction success \| page=(\d+) \| idx=(\d+) \| extractor=(\w+) \| content_length=(\d+)",
        re.IGNORECASE
    )
    extract_fail_pattern = re.compile(
        r"Image extraction failure \| page=(\d+) \| idx=(\d+) \| extractor=(\w+)",
        re.IGNORECASE
    )

    image_details = {}

    for log_line in log_capture.records:
        # Match classification success
        m = class_pattern.search(log_line)
        if m:
            page, idx, img_type, conf, reason = m.groups()
            key = (int(page), int(idx))
            if key not in image_details:
                image_details[key] = {}
            image_details[key].update({
                "page_number": int(page),
                "image_index": int(idx),
                "image_type": img_type,
                "confidence": float(conf),
                "reason": reason,
                "classification_status": "success"
            })
            continue

        # Match classification failure
        m = class_fail_pattern.search(log_line)
        if m:
            page, idx, err = m.groups()
            key = (int(page), int(idx))
            if key not in image_details:
                image_details[key] = {}
            image_details[key].update({
                "page_number": int(page),
                "image_index": int(idx),
                "image_type": "UNKNOWN",
                "confidence": 0.0,
                "reason": f"Failed: {err}",
                "classification_status": "failed"
            })
            continue

        # Match extraction success
        m = extract_pattern.search(log_line)
        if m:
            page, idx, extractor, length = m.groups()
            key = (int(page), int(idx))
            if key not in image_details:
                image_details[key] = {}
            image_details[key].update({
                "extractor_selected": extractor,
                "content_length": int(length),
                "extraction_status": "success"
            })
            continue

        # Match extraction failure
        m = extract_fail_pattern.search(log_line)
        if m:
            page, idx, extractor = m.groups()
            key = (int(page), int(idx))
            if key not in image_details:
                image_details[key] = {}
            image_details[key].update({
                "extractor_selected": extractor,
                "content_length": 0,
                "extraction_status": "failed"
            })
            continue

    # 5. Query SQLite for the actual chunk contents
    db_chunks = []
    if ingestion_success and new_doc_id:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number, sibling_order",
                (new_doc_id,)
            ).fetchall()
            db_chunks = [dict(r) for r in rows]
            conn.close()
        except Exception as db_err:
            print(f"[ERROR] Failed to query database: {db_err}")

    # Map database chunks of type "image" to the captured image log entries
    image_chunks = [c for c in db_chunks if c["chunk_type"] == "image"]
    
    # Sort log details by page_number, then image_index
    sorted_keys = sorted(image_details.keys())
    
    validated_images = []
    for key in sorted_keys:
        log_info = image_details[key]
        
        # Match with the corresponding DB chunk sequentially on the same page
        matching_chunk = None
        for chunk in image_chunks:
            if chunk["page_number"] == log_info["page_number"]:
                # Let's count how many image chunks on this page before this one
                page_chunks = [c for c in image_chunks if c["page_number"] == log_info["page_number"]]
                try:
                    idx_on_page = page_chunks.index(chunk)
                except ValueError:
                    idx_on_page = -1
                if idx_on_page == log_info["image_index"]:
                    matching_chunk = chunk
                    break
        
        content = matching_chunk["content"] if matching_chunk else "[Not found in SQLite]"
        chunk_id = matching_chunk["id"] if matching_chunk else "[N/A]"
        
        validated_images.append({
            "page_number": log_info["page_number"],
            "image_index": log_info["image_index"],
            "image_type": log_info.get("image_type", "UNKNOWN"),
            "confidence": log_info.get("confidence", 0.0),
            "reason": log_info.get("reason", ""),
            "extractor_selected": log_info.get("extractor_selected", "None"),
            "extraction_status": log_info.get("extraction_status", "failed"),
            "chunk_id": chunk_id,
            "chunk_length": len(content) if matching_chunk else 0,
            "generated_chunk_content": content,
            "vector_insertion_status": "success" if (ingestion_success and matching_chunk) else "failed"
        })

    # Summary calculations
    images_detected = len(validated_images)
    success_count = sum(1 for img in validated_images if img["extraction_status"] == "success")

    # 6. Save validation report as JSON
    report_data = {
        "report_metadata": {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_pdf": filename,
            "ingestion_success": ingestion_success,
            "ingestion_message": msg,
            "execution_time_seconds": round(t_elapsed, 2),
            "images_detected": images_detected,
            "images_successfully_extracted": success_count,
            "total_chunks_created": len(db_chunks),
            "error": error_occurred
        },
        "images_validation": validated_images
    }

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Validation report successfully saved to: {report_path}")
    except Exception as save_err:
        print(f"[ERROR] Failed to save report JSON: {save_err}")

    # 7. Print validation summary in console
    print("\n" + "=" * 60)
    print("                 VALIDATION TEST SUMMARY")
    print("=" * 60)
    print(f"Target PDF         : {filename}")
    print(f"Ingestion Status   : {'SUCCESS' if ingestion_success else 'FAILED'}")
    print(f"Total Chunks       : {len(db_chunks)}")
    print(f"Images Detected    : {images_detected}")
    print(f"Images Extracted   : {success_count} / {images_detected}")
    print(f"Execution Time     : {t_elapsed:.2f}s")
    print("-" * 60)
    
    for img in validated_images:
        print(f"\n* Page {img['page_number']}, Image {img['image_index']}:")
        print(f"  Classification   : {img['image_type']} (Confidence: {img['confidence']:.4f})")
        print(f"  Reason           : {img['reason']}")
        print(f"  Extractor        : {img['extractor_selected']}")
        print(f"  Extraction       : {img['extraction_status'].upper()}")
        print(f"  Chunk Length     : {img['chunk_length']} characters")
        print(f"  Vector Indexing  : {img['vector_insertion_status'].upper()}")
        print("  Snippet          :")
        snippet = img['generated_chunk_content'][:180].replace('\n', ' ')
        safe_print(f"    \"{snippet}...\"")
    print("=" * 60)

if __name__ == "__main__":
    main()
