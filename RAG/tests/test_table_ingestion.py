import os
import sys
import uuid
import json
import time
import shutil
import sqlite3
from typing import List, Dict, Any

# Ensure root folder is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from RAG.ingestion import process_pdf
from RAG.db import init_db
from RAG.vector_store import init_vector_store, get_qdrant_client, COLLECTION_NAME

def generate_table_test_pdf(pdf_path: str):
    """Generates a PDF using reportlab containing all 6 table types on different pages."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Page 1: TABLE_SIMPLE
    story.append(Paragraph("<b>Page 1: Simple Employee Directory</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data1 = [
        ["First Name", "Last Name", "Role", "Department"],
        ["John", "Doe", "Engineer", "Tech"],
        ["Jane", "Smith", "Manager", "Sales"],
        ["Bob", "Johnson", "Recruiter", "HR"]
    ]
    t1 = Table(data1)
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t1)
    story.append(PageBreak())

    # Page 2: TABLE_COMPARISON
    story.append(Paragraph("<b>Page 2: Object Detection Model Comparison Matrix</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data2 = [
        ["Model Name", "Precision", "Recall", "Latency", "Parameters", "Key Use Case"],
        ["YOLOv8", "0.925", "0.880", "12.4 ms", "3.2 M", "Real-time edge tracking"],
        ["YOLOX", "0.941", "0.908", "15.1 ms", "9.0 M", "Precise document layout inspection"],
        ["YOLOv7", "0.912", "0.865", "11.0 ms", "6.2 M", "High fps video analytics"]
    ]
    t2 = Table(data2)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t2)
    story.append(PageBreak())

    # Page 3: TABLE_FINANCIAL
    story.append(Paragraph("<b>Page 3: FY2025 Corporate Revenue Report</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data3 = [
        ["Financial Category", "Q1 FY2025", "Q2 FY2025", "Q3 FY2025", "Q4 FY2025 (Projected)"],
        ["Gross Revenue", "$120.0M", "$150.0M", "$180.0M", "$210.0M"],
        ["Operating Expenses", "$80.0M", "$90.0M", "$100.0M", "$110.0M"],
        ["Net profit", "$40.0M", "$60.0M", "$80.0M", "$100.0M"]
    ]
    t3 = Table(data3)
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t3)
    story.append(PageBreak())

    # Page 4: TABLE_STATISTICAL
    story.append(Paragraph("<b>Page 4: Machine Learning Model Benchmark Metrics</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data4 = [
        ["Variable Evaluated", "Mean Score", "Standard Deviation", "P-value", "Sample Count (N)"],
        ["Baseline Group A", "0.784", "0.045", "Reference", "150"],
        ["Experimental Group B", "0.812", "0.038", "0.024 (Significant)", "150"],
        ["Experimental Group C", "0.835", "0.031", "<0.001 (Highly Significant)", "150"]
    ]
    t4 = Table(data4)
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t4)
    story.append(PageBreak())

    # Page 5: TABLE_TIMESERIES
    story.append(Paragraph("<b>Page 5: Active Users Monthly Chronological Trend</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data5 = [
        ["Chronological Month", "Monthly Active Users", "Acquisition Cost (CAC)", "Churn Rate"],
        ["January 2025", "82,000", "$42.50", "4.2%"],
        ["February 2025", "85,500", "$40.10", "4.0%"],
        ["March 2025", "91,200", "$38.50", "3.8%"],
        ["April 2025", "98,400", "$36.00", "3.5%"]
    ]
    t5 = Table(data5)
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t5)
    story.append(PageBreak())

    # Page 6: TABLE_UNKNOWN
    story.append(Paragraph("<b>Page 6: Layout Element Placeholder</b>", styles['Heading1']))
    story.append(Spacer(1, 10))
    data6 = [
        ["This is a simple single-cell paragraph block that acts as a structural placeholder rather than a real table."],
        ["Note: No headers or columns are defined here."]
    ]
    t6 = Table(data6)
    t6.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t6)

    doc.build(story)

def main():
    print("=" * 60)
    print("      TABLE INTEED-TO-END INGESTION VALIDATION")
    print("=" * 60)

    # 1. Setup paths
    data_dir = os.path.join(BASE_DIR, "RAG", "data")
    pdf_path = os.path.join(data_dir, "test_all_tables.pdf")
    db_path = os.path.join(data_dir, "rag_tool_test.db")
    vector_db_path = os.path.join(data_dir, "qdrant_test")
    report_path = os.path.join(BASE_DIR, "outputs", "table_ingestion_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Clean up any stale test assets
    for path in [pdf_path, db_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)

    # 2. Generate PDF and init database/vector store
    print("Generating multi-page tables PDF...")
    generate_table_test_pdf(pdf_path)
    print(f"PDF generated successfully: {pdf_path}")

    init_db(db_path)
    init_vector_store(vector_db_path)

    # 3. Perform ingestion
    print("\nIngesting PDF file...")
    t0 = time.perf_counter()
    doc_id, msg = process_pdf(pdf_path, db_path, vector_db_path)
    elapsed = time.perf_counter() - t0
    print(f"Ingestion Finished in {elapsed:.2f}s: {msg}")

    # 4. Verify SQLite write
    print("\nVerifying SQLite chunks insertion...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, page_number, chunk_type, content, sibling_order FROM chunks WHERE chunk_type = 'table'")
    db_rows = cursor.fetchall()
    conn.close()

    print(f"Found {len(db_rows)} table chunks in SQLite database.")
    sqlite_verified = len(db_rows) >= 5 # 5 expected structured tables

    # 5. Verify Qdrant write
    print("\nVerifying Qdrant vector store indexing & metadata...")
    qdrant_client = get_qdrant_client(vector_db_path)
    scroll_result = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    qdrant_client.close()

    q_points = scroll_result[0]
    table_points = [p for p in q_points if p.payload.get("metadata", {}).get("chunk_type") == "table"]
    print(f"Found {len(table_points)} table points indexed in Qdrant.")
    qdrant_verified = len(table_points) >= 5

    # 6. Extract results & compile report
    report_items = []
    metadata_verified = True

    print("\nExtraction Summary:")
    print(f"{'Page':<6} | {'Classified Table Type':<18} | {'Confidence':<10} | {'Extractor Selected'}")
    print("-" * 75)

    # Let's map qdrant payloads to report
    for pt in table_points:
        meta = pt.payload.get("metadata", {})
        page = meta.get("page_number")
        t_type = meta.get("table_type")
        conf = meta.get("classification_confidence")
        reason = meta.get("classification_reason")
        
        # Check if they exist
        if not t_type or conf is None:
            metadata_verified = False
            
        # Match with SQLite contents to find extractor used
        # (our logs will show it, but we can also infer based on type)
        extractor = "unknown"
        if t_type == "TABLE_SIMPLE":
            extractor = "extract_simple_table"
        elif t_type == "TABLE_COMPARISON":
            extractor = "extract_comparison_table"
        elif t_type == "TABLE_FINANCIAL":
            extractor = "extract_financial_table"
        elif t_type == "TABLE_STATISTICAL":
            extractor = "extract_statistical_table"
        elif t_type == "TABLE_TIMESERIES":
            extractor = "extract_timeseries_table"
        elif t_type == "TABLE_UNKNOWN":
            extractor = "extract_unknown_table"

        content_len = len(pt.payload.get("page_content", ""))
        print(f"{page:<6} | {str(t_type):<18} | {conf:<10.4f} | {extractor:<30} | length={content_len} chars")

        report_items.append({
            "page_number": page,
            "table_type": t_type,
            "confidence": conf,
            "reason": reason,
            "extractor_selected": extractor,
            "chunk_length": content_len,
            "qdrant_point_id": pt.id
        })

    # Save validation report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "sqlite_verified": sqlite_verified,
        "qdrant_verified": qdrant_verified,
        "metadata_verified": metadata_verified,
        "ingested_tables": sorted(report_items, key=lambda x: x["page_number"])
    }
    
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=2)
    print(f"\nValidation report saved to: {report_path}")

    # Clean up test files
    print("\nCleaning up test databases and collections...")
    for path in [pdf_path, db_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)
    print("Cleanup complete.")

    # Assertions success check
    success = sqlite_verified and qdrant_verified and metadata_verified
    print("\n" + "=" * 60)
    if success:
        print("  SUCCESS: Ingestion, SQLite, Qdrant, and Metadata Verified!")
    else:
        print("  FAILURE: Missing table index points or payload metadata.")
    print("=" * 60)

if __name__ == "__main__":
    main()
