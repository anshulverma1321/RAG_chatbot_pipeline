"""
Table Retrieval Validation Suite
=================================
Proves that Table Intelligence improves retrieval quality and answer correctness.

Workflow (per run):
  1. Generate test_all_tables.pdf with all 6 table types.
  2. Ingest PDF into an isolated test database + vector store.
  3. For every table type, run 3+ targeted queries.
  4. For every query:
       a. Retrieve top chunks via vector search.
       b. Extract metadata (table_type, confidence, reason from Qdrant payload).
       c. Generate a grounded answer using execute_rag_query().
       d. Evaluate answer quality with Gemini LLM-as-a-judge.
  5. Persist full results to outputs/table_retrieval_validation_report.json.
  6. Clean up the isolated test database + vector store.
"""

import os
import sys
import re
import json
import time
import shutil
import sqlite3

# Force UTF-8 output on Windows consoles (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Project root resolution and environment loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from RAG.ingestion import process_pdf
from RAG.db import init_db, list_documents
from RAG.vector_store import init_vector_store, get_qdrant_client, search_vectors, COLLECTION_NAME
from RAG.query_engine import execute_rag_query, get_query_embedding


# ---------------------------------------------------------------------------
# Test-query catalog — 3 questions per table type (6 types × 3 = 18 queries)
# ---------------------------------------------------------------------------
TABLE_QUERIES = [
    # ── TABLE_SIMPLE ────────────────────────────────────────────────────────
    {
        "table_type": "TABLE_SIMPLE",
        "page_number": 1,
        "questions": [
            "List all employees in the Simple Employee Directory.",
            "Which department does Jane Smith belong to?",
            "What role does Bob Johnson hold in the organisation?",
        ],
    },
    # ── TABLE_COMPARISON ────────────────────────────────────────────────────
    {
        "table_type": "TABLE_COMPARISON",
        "page_number": 2,
        "questions": [
            "Which object detection model achieved the highest precision score?",
            "Compare the precision and recall values across YOLOv8, YOLOX, and YOLOv7.",
            "What are the key differences between YOLOv8 and YOLOX in terms of parameters and latency?",
        ],
    },
    # ── TABLE_FINANCIAL ─────────────────────────────────────────────────────
    {
        "table_type": "TABLE_FINANCIAL",
        "page_number": 3,
        "questions": [
            "What was the highest quarterly gross revenue reported in FY2025?",
            "In which quarter was net profit the largest?",
            "Describe the overall financial trend across Q1 to Q4 FY2025 in terms of revenue and expenses.",
        ],
    },
    # ── TABLE_STATISTICAL ───────────────────────────────────────────────────
    {
        "table_type": "TABLE_STATISTICAL",
        "page_number": 4,
        "questions": [
            "Which experimental group achieved the highest mean score in the benchmark?",
            "What were the p-values for experimental group B and group C and what do they indicate?",
            "Summarise the key statistical conclusions from the machine learning benchmark table.",
        ],
    },
    # ── TABLE_TIMESERIES ────────────────────────────────────────────────────
    {
        "table_type": "TABLE_TIMESERIES",
        "page_number": 5,
        "questions": [
            "Which month had the highest number of monthly active users?",
            "What trend is visible in the customer acquisition cost (CAC) from January to April 2025?",
            "Were there any significant drops or improvements in churn rate across the tracked months?",
        ],
    },
    # ── TABLE_UNKNOWN ───────────────────────────────────────────────────────
    {
        "table_type": "TABLE_UNKNOWN",
        "page_number": 6,
        "questions": [
            "What is described on page 6 of the document?",
            "What structural role does the placeholder element on page 6 serve?",
            "Summarise the textual content found on the sixth page of the document.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper: Generate test PDF
# ---------------------------------------------------------------------------
def generate_table_test_pdf(pdf_path: str):
    """Generates a 6-page PDF with one table type per page using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph, Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Page 1 — TABLE_SIMPLE
    story.append(Paragraph("<b>Page 1: Simple Employee Directory</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["First Name", "Last Name", "Role", "Department"],
            ["John", "Doe", "Engineer", "Tech"],
            ["Jane", "Smith", "Manager", "Sales"],
            ["Bob", "Johnson", "Recruiter", "HR"],
        ],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))
    story.append(PageBreak())

    # Page 2 — TABLE_COMPARISON
    story.append(Paragraph("<b>Page 2: Object Detection Model Comparison Matrix</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["Model Name", "Precision", "Recall", "Latency", "Parameters", "Key Use Case"],
            ["YOLOv8", "0.925", "0.880", "12.4 ms", "3.2 M", "Real-time edge tracking"],
            ["YOLOX",  "0.941", "0.908", "15.1 ms", "9.0 M", "Precise document layout inspection"],
            ["YOLOv7", "0.912", "0.865", "11.0 ms", "6.2 M", "High fps video analytics"],
        ],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))
    story.append(PageBreak())

    # Page 3 — TABLE_FINANCIAL
    story.append(Paragraph("<b>Page 3: FY2025 Corporate Revenue Report</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["Financial Category", "Q1 FY2025", "Q2 FY2025", "Q3 FY2025", "Q4 FY2025 (Projected)"],
            ["Gross Revenue",      "$120.0M",   "$150.0M",   "$180.0M",   "$210.0M"],
            ["Operating Expenses", "$80.0M",    "$90.0M",    "$100.0M",   "$110.0M"],
            ["Net profit",         "$40.0M",    "$60.0M",    "$80.0M",    "$100.0M"],
        ],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))
    story.append(PageBreak())

    # Page 4 — TABLE_STATISTICAL
    story.append(Paragraph("<b>Page 4: Machine Learning Model Benchmark Metrics</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["Variable Evaluated",   "Mean Score", "Standard Deviation", "P-value",                     "Sample Count (N)"],
            ["Baseline Group A",     "0.784",      "0.045",              "Reference",                   "150"],
            ["Experimental Group B", "0.812",      "0.038",              "0.024 (Significant)",          "150"],
            ["Experimental Group C", "0.835",      "0.031",              "<0.001 (Highly Significant)", "150"],
        ],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))
    story.append(PageBreak())

    # Page 5 — TABLE_TIMESERIES
    story.append(Paragraph("<b>Page 5: Active Users Monthly Chronological Trend</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["Chronological Month", "Monthly Active Users", "Acquisition Cost (CAC)", "Churn Rate"],
            ["January 2025",        "82,000",               "$42.50",                 "4.2%"],
            ["February 2025",       "85,500",               "$40.10",                 "4.0%"],
            ["March 2025",          "91,200",               "$38.50",                 "3.8%"],
            ["April 2025",          "98,400",               "$36.00",                 "3.5%"],
        ],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))
    story.append(PageBreak())

    # Page 6 — TABLE_UNKNOWN
    story.append(Paragraph("<b>Page 6: Layout Element Placeholder</b>", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Table(
        [
            ["This is a simple single-cell paragraph block that acts as a structural placeholder rather than a real table."],
            ["Note: No headers or columns are defined here."],
        ],
        style=TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]),
    ))

    doc.build(story)


# ---------------------------------------------------------------------------
# Helper: LLM-as-a-Judge evaluation
# ---------------------------------------------------------------------------
def evaluate_answer_quality(query: str, context: str, answer: str) -> dict:
    """Calls Gemini to score an answer on groundedness, completeness, correctness (1-5)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    prompt = (
        "You are an expert RAG system evaluator. Assess the quality of the generated answer "
        "based on the query and the retrieved context.\n\n"
        f"Query:\n{query}\n\n"
        f"Retrieved Context (truncated to 1500 chars):\n{context[:1500]}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "Respond with a JSON object ONLY — no markdown fences, no prose:\n"
        "{\n"
        "  \"groundedness\": int,    // 1-5: is the answer derived ONLY from the context?\n"
        "  \"completeness\": int,    // 1-5: does it fully answer every part of the query?\n"
        "  \"correctness\": int,     // 1-5: is it factually accurate relative to the context?\n"
        "  \"justification\": \"string\" // 1-2 sentence rationale\n"
        "}"
    )
    try:
        response = chat.invoke(prompt)
        text = response.content
        if isinstance(text, list):
            parts = []
            for p in text:
                if isinstance(p, dict) and "text" in p:
                    parts.append(p["text"])
                elif isinstance(p, str):
                    parts.append(p)
            text = "".join(parts)
        text = text.strip()
        # Strip any accidental markdown code fences
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        return {
            "groundedness": 0,
            "completeness": 0,
            "correctness": 0,
            "justification": f"Evaluation failed: {e}",
        }


# ---------------------------------------------------------------------------
# Helper: safe console output (handles Windows cp1252 issues)
# ---------------------------------------------------------------------------
def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


# ---------------------------------------------------------------------------
# Helper: infer table_type from chunk content headers (fallback)
# ---------------------------------------------------------------------------
def infer_table_type_from_content(content: str) -> str:
    if "Financial Data Analysis" in content or "TABLE_FINANCIAL" in content:
        return "TABLE_FINANCIAL"
    if "Chronological Trend" in content or "TABLE_TIMESERIES" in content:
        return "TABLE_TIMESERIES"
    if "Statistical Findings" in content or "TABLE_STATISTICAL" in content:
        return "TABLE_STATISTICAL"
    if "Comparison Table" in content or "TABLE_COMPARISON" in content:
        return "TABLE_COMPARISON"
    if "Simple Table" in content or "TABLE_SIMPLE" in content:
        return "TABLE_SIMPLE"
    if "Unclassified Table" in content or "TABLE_UNKNOWN" in content:
        return "TABLE_UNKNOWN"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("            TABLE RETRIEVAL VALIDATION SUITE")
    print("=" * 70)

    # ── Path setup ────────────────────────────────────────────────────────
    data_dir        = os.path.join(BASE_DIR, "RAG", "data")
    pdf_path        = os.path.join(data_dir, "test_all_tables_retrieval.pdf")
    db_path         = os.path.join(data_dir, "rag_retrieval_test.db")
    vector_db_path  = os.path.join(data_dir, "qdrant_retrieval_test")
    report_path     = os.path.join(BASE_DIR, "outputs", "table_retrieval_validation_report.json")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # ── Clean up any stale test assets from a previous run ───────────────
    for path in [pdf_path, db_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Ingestion
    # ════════════════════════════════════════════════════════════════════════
    print("\n[PHASE 1] Generating test PDF...")
    generate_table_test_pdf(pdf_path)
    print(f"  PDF ready: {pdf_path}")

    print("[PHASE 1] Initialising isolated database and vector store...")
    init_db(db_path)
    init_vector_store(vector_db_path)

    print("[PHASE 1] Ingesting PDF...")
    t_ingest_start = time.perf_counter()
    doc_id, ingest_msg = process_pdf(pdf_path, db_path, vector_db_path)
    t_ingest_elapsed = time.perf_counter() - t_ingest_start
    print(f"  {ingest_msg}")
    print(f"  Ingestion time : {t_ingest_elapsed:.2f}s")

    # Retrieve the ingested document IDs (just the one we uploaded)
    docs    = list_documents(db_path)
    doc_ids = [d["id"] for d in docs]
    if not doc_ids:
        print("[ERROR] Ingestion produced no document records. Aborting.")
        sys.exit(1)

    # ── Collect Qdrant metadata for every table chunk ────────────────────
    qdrant_client = get_qdrant_client(vector_db_path)
    scroll_result = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    qdrant_client.close()

    all_points = scroll_result[0]
    # Build a map: chunk_content_prefix -> qdrant metadata
    chunk_meta_map: dict = {}
    for pt in all_points:
        meta = pt.payload.get("metadata", {})
        content = pt.payload.get("page_content", "")
        chunk_meta_map[pt.id] = {
            "table_type":               meta.get("table_type", "N/A"),
            "classification_confidence": meta.get("classification_confidence", None),
            "classification_reason":    meta.get("classification_reason", "N/A"),
            "chunk_type":               meta.get("chunk_type", "N/A"),
            "page_number":              meta.get("page_number", None),
            "filename":                 meta.get("filename", "N/A"),
            "content_prefix":           content[:120],
        }

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Retrieval + Answer Generation + Evaluation
    # ════════════════════════════════════════════════════════════════════════
    print("\n[PHASE 2] Running retrieval and answer generation...\n")

    all_results: list = []
    grand_total_q = 0
    grand_scores  = {"groundedness": 0, "completeness": 0, "correctness": 0}

    for group in TABLE_QUERIES:
        table_type  = group["table_type"]
        page_num    = group["page_number"]
        questions   = group["questions"]

        print("-" * 70)
        print(f"  TABLE TYPE : {table_type}  (expected on page {page_num})")
        print("-" * 70)

        group_results = []

        for q_idx, query in enumerate(questions, 1):
            safe_print(f"\n  [{q_idx}/{len(questions)}] Query: \"{query}\"")

            t_q_start = time.perf_counter()

            # ── Vector retrieval ─────────────────────────────────────────
            query_vector = get_query_embedding(query)
            raw_hits: list = []
            for did in doc_ids:
                hits = search_vectors(vector_db_path, query_vector, [did], top_k=5)
                raw_hits.extend(hits)
            raw_hits.sort(key=lambda x: x["score"], reverse=True)
            top_hits = raw_hits[:5]

            # ── Parse chunk metadata from Qdrant payload ─────────────────
            retrieved_chunks = []
            context_parts    = []

            for hit in top_hits:
                chunk_id   = hit["id"]
                score      = round(hit["score"], 4)
                content    = hit["content"]
                qdrant_pay = hit["payload"]

                # Prefer live Qdrant metadata; fall back to content header inference
                meta_entry       = chunk_meta_map.get(chunk_id, {})
                tbl_type_payload = meta_entry.get("table_type", "N/A")
                conf_payload     = meta_entry.get("classification_confidence")
                reason_payload   = meta_entry.get("classification_reason", "N/A")
                chunk_type_pay   = meta_entry.get("chunk_type") or qdrant_pay.get("chunk_type", "N/A")
                page_pay         = meta_entry.get("page_number") or qdrant_pay.get("page_number")
                fname_pay        = meta_entry.get("filename") or qdrant_pay.get("filename", "N/A")

                # Content-based inference if metadata absent
                if tbl_type_payload == "N/A":
                    tbl_type_payload = infer_table_type_from_content(content)

                retrieved_chunks.append({
                    "chunk_id":              chunk_id,
                    "retrieval_score":       score,
                    "chunk_type":            chunk_type_pay,
                    "table_type":            tbl_type_payload,
                    "classification_confidence": conf_payload,
                    "classification_reason": reason_payload,
                    "page_number":           page_pay,
                    "filename":              fname_pay,
                    "content_preview":       content[:200].replace("\n", " ") + "...",
                })
                context_parts.append(content)

            retrieved_context = "\n\n".join(context_parts)

            # ── Grounded answer generation ────────────────────────────────
            try:
                answer = execute_rag_query(query, db_path, vector_db_path, document_ids=doc_ids)
                generation_ok = True
            except Exception as gen_err:
                answer = f"[Generation error: {gen_err}]"
                generation_ok = False

            t_q_elapsed = time.perf_counter() - t_q_start

            # ── LLM-as-a-Judge quality evaluation ─────────────────────────
            if generation_ok and retrieved_context:
                eval_metrics = evaluate_answer_quality(query, retrieved_context, answer)
            else:
                eval_metrics = {
                    "groundedness": 0,
                    "completeness": 0,
                    "correctness": 0,
                    "justification": "Skipped — generation failed.",
                }

            # Accumulate for summary averages
            grand_total_q += 1
            grand_scores["groundedness"] += eval_metrics.get("groundedness", 0)
            grand_scores["completeness"] += eval_metrics.get("completeness", 0)
            grand_scores["correctness"]  += eval_metrics.get("correctness",  0)

            # ── Console print ─────────────────────────────────────────────
            print(f"     Chunks retrieved   : {len(retrieved_chunks)}")
            if retrieved_chunks:
                best = retrieved_chunks[0]
                print(f"     Top chunk score    : {best['retrieval_score']}")
                print(f"     Top chunk type     : {best['table_type']} (page {best['page_number']})")
            print(f"     Groundedness       : {eval_metrics.get('groundedness')}/5")
            print(f"     Completeness       : {eval_metrics.get('completeness')}/5")
            print(f"     Correctness        : {eval_metrics.get('correctness')}/5")
            print(f"     Total query time   : {t_q_elapsed:.2f}s")

            # ── Grounding source (first citation in answer) ───────────────
            citation_match = re.search(r"\[(.+?)\]", answer)
            grounding_source = citation_match.group(0) if citation_match else "No explicit citation found."

            group_results.append({
                "query_index":       q_idx,
                "query":             query,
                "execution_time_s":  round(t_q_elapsed, 2),
                "retrieved_chunks":  retrieved_chunks,
                "final_answer":      answer,
                "grounding_source":  grounding_source,
                "evaluation":        eval_metrics,
            })

        # Per-group average
        grp_ground = sum(r["evaluation"].get("groundedness", 0) for r in group_results) / len(group_results)
        grp_compl  = sum(r["evaluation"].get("completeness",  0) for r in group_results) / len(group_results)
        grp_corr   = sum(r["evaluation"].get("correctness",   0) for r in group_results) / len(group_results)

        print(f"\n  >> {table_type} group averages -- "
              f"G: {grp_ground:.2f}  C: {grp_compl:.2f}  Cr: {grp_corr:.2f}")

        all_results.append({
            "table_type":       table_type,
            "expected_page":    page_num,
            "group_averages":   {
                "groundedness": round(grp_ground, 2),
                "completeness": round(grp_compl, 2),
                "correctness":  round(grp_corr, 2),
            },
            "queries": group_results,
        })

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Save report
    # ════════════════════════════════════════════════════════════════════════
    overall_ground = grand_scores["groundedness"] / grand_total_q if grand_total_q else 0
    overall_compl  = grand_scores["completeness"]  / grand_total_q if grand_total_q else 0
    overall_corr   = grand_scores["correctness"]   / grand_total_q if grand_total_q else 0

    report = {
        "report_metadata": {
            "test_timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_source":             os.path.basename(pdf_path),
            "total_table_types":      len(TABLE_QUERIES),
            "total_queries":          grand_total_q,
            "ingestion_time_seconds": round(t_ingest_elapsed, 2),
            "evaluation_engine":      "Gemini-3.1-Flash-Lite (LLM-as-a-judge)",
        },
        "overall_quality_card": {
            "avg_groundedness": round(overall_ground, 2),
            "avg_completeness": round(overall_compl,  2),
            "avg_correctness":  round(overall_corr,   2),
        },
        "table_type_results": all_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Report saved to: {report_path}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Cleanup
    # ════════════════════════════════════════════════════════════════════════
    print("\n[PHASE 4] Cleaning up isolated test databases...")
    for path in [pdf_path, db_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)
    print("  Cleanup complete.")

    # ════════════════════════════════════════════════════════════════════════
    # Final quality card
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("               TABLE RETRIEVAL QUALITY CARD")
    print("=" * 70)
    print(f"  Total Queries Evaluated : {grand_total_q}")
    print(f"  Table Types Covered     : {len(TABLE_QUERIES)}")
    print(f"  Avg Groundedness        : {overall_ground:.2f} / 5.0")
    print(f"  Avg Completeness        : {overall_compl:.2f} / 5.0")
    print(f"  Avg Correctness         : {overall_corr:.2f} / 5.0")
    print("=" * 70)


if __name__ == "__main__":
    main()
