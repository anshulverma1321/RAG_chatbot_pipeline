import os
import sys
import json
import time
import re
import sqlite3

# Resolve project root and make it importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from RAG.query_engine import execute_rag_query, get_query_embedding
from RAG.vector_store import search_vectors

def load_image_type_map(report_path):
    """Loads image chunk ID to image type mapping from the ingestion report."""
    mapping = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for img in data.get("images_validation", []):
                chunk_id = img.get("chunk_id")
                img_type = img.get("image_type")
                if chunk_id and img_type:
                    mapping[chunk_id] = img_type
        except Exception as e:
            print(f"[WARN] Failed to load ingestion report: {e}")
    return mapping

def evaluate_answer_quality(query: str, retrieved_context: str, answer: str) -> dict:
    """Uses Gemini to evaluate final answer quality (groundedness, completeness, correctness)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    
    prompt = (
        "You are an expert RAG system evaluator. Assess the quality of the generated answer "
        "based on the query and the retrieved context.\n\n"
        f"Query:\n{query}\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "Provide a JSON response with the following format:\n"
        "{\n"
        "  \"groundedness\": int,  // score 1 to 5 (is the answer derived ONLY from context?)\n"
        "  \"completeness\": int,  // score 1 to 5 (does it fully answer the query?)\n"
        "  \"correctness\": int,   // score 1 to 5 (is it factually accurate relative to context?)\n"
        "  \"justification\": \"string\" // short rationale for the scores\n"
        "}"
    )
    try:
        response = chat.invoke(prompt)
        text = response.content
        if isinstance(text, list):
            text_parts = []
            for part in text:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            text = "".join(text_parts)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
        return json.loads(text)
    except Exception as eval_err:
        return {
            "groundedness": 0,
            "completeness": 0,
            "correctness": 0,
            "justification": f"Evaluation failed: {eval_err}"
        }

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
    ingest_report_path = os.path.join(BASE_DIR, "outputs", "image_ingestion_report.json")
    report_path = os.path.join(BASE_DIR, "outputs", "retrieval_validation_report.json")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    print("=" * 60)
    print("      MULTIMODAL RETRIEVAL VALIDATION SUITE")
    print("=" * 60)
    print(f"Database Path  : {db_path}")
    print(f"Vector DB Path : {vector_db_path}")
    print(f"Report Path    : {report_path}\n")

    # Load chunk-id -> image-type mapping
    image_type_map = load_image_type_map(ingest_report_path)
    print(f"Loaded {len(image_type_map)} image mappings from ingestion report.\n")

    # Define test queries for the 5 categories
    test_queries = [
        {
            "category": "Chart Values",
            "query": "What Cartesian plane and description correspond to Roll in the degrees of freedom mapping?"
        },
        {
            "category": "Architecture Diagrams",
            "query": "What are the internal sensors of the IMU and what functions do they execute?"
        },
        {
            "category": "Infographic Content",
            "query": "Explain the Right-Hand Rule and pose calculations for yaw, roll, and pitch."
        },
        {
            "category": "OCR Text Images",
            "query": "What Creative Commons license and suggested citation is recommended for the World Health Organization document?"
        },
        {
            "category": "Natural Photographs",
            "query": "What labeled pins are visible on the physical MPU-6050 board photograph?"
        }
    ]

    from RAG.db import list_documents
    try:
        docs = list_documents(db_path)
        doc_ids = [d['id'] for d in docs]
    except Exception as e:
        print(f"[ERROR] Failed to list documents: {e}")
        sys.exit(1)

    if not doc_ids:
        print("[ERROR] No documents ingested. Please ingest test_images.pdf and documentation.pdf first.")
        sys.exit(1)

    validation_results = []

    for idx, q_info in enumerate(test_queries, 1):
        category = q_info["category"]
        query = q_info["query"]

        print(f"[{idx}/5] Processing Category: {category}")
        print(f"    Query: \"{query}\"")

        # 1. Retrieve top chunks and scores
        t_start = time.perf_counter()
        try:
            query_vector = get_query_embedding(query)
            
            all_hits = []
            for doc_id in doc_ids:
                hits = search_vectors(vector_db_path, query_vector, [doc_id], top_k=3)
                all_hits.extend(hits)
                
            all_hits.sort(key=lambda x: x['score'], reverse=True)
            top_hits = all_hits[:5]
        except Exception as ret_err:
            print(f"    [ERROR] Retrieval failed: {ret_err}")
            top_hits = []

        # Parse retrieved chunks
        retrieved_chunks = []
        retrieved_image_types = set()
        context_parts = []

        for hit in top_hits:
            chunk_id = hit["id"]
            score = hit["score"]
            content = hit["content"]
            payload = hit["payload"]
            chunk_type = payload.get("chunk_type")

            image_type = "N/A"
            if chunk_type == "image":
                image_type = image_type_map.get(chunk_id, "UNKNOWN")
                if image_type == "UNKNOWN":
                    # Fallback parse from headers
                    if "Diagram Structure" in content:
                        image_type = "DIAGRAM"
                    elif "Natural Image Description" in content:
                        image_type = "NATURAL_IMAGE"
                    elif "Chart Knowledge Extraction" in content:
                        image_type = "CHART"
                    elif "Document Text Knowledge Extraction" in content:
                        image_type = "TEXT_IMAGE"
                    elif "Mixed Image" in content or "Technical Analysis" in content:
                        image_type = "MIXED"
                retrieved_image_types.add(image_type)

            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "chunk_type": chunk_type,
                "image_type": image_type,
                "filename": payload.get("filename"),
                "page_number": payload.get("page_number"),
                "content_preview": content[:180].replace('\n', ' ') + "..."
            })
            context_parts.append(content)

        retrieved_context = "\n\n".join(context_parts)

        # 2. Run grounded generation
        try:
            answer = execute_rag_query(query, db_path, vector_db_path)
            generation_success = True
        except Exception as gen_err:
            answer = f"Generation failed: {gen_err}"
            generation_success = False

        t_elapsed = time.perf_counter() - t_start

        # 3. Evaluate answer quality
        eval_metrics = {
            "groundedness": 0,
            "completeness": 0,
            "correctness": 0,
            "justification": "N/A"
        }
        if generation_success and retrieved_context:
            eval_metrics = evaluate_answer_quality(query, retrieved_context, answer)

        validation_results.append({
            "category": category,
            "query": query,
            "execution_time_seconds": round(t_elapsed, 2),
            "retrieved_image_types": list(retrieved_image_types),
            "retrieved_chunks": retrieved_chunks,
            "final_answer": answer,
            "evaluation": eval_metrics
        })

        print(f"    Chunks Retrieved : {len(retrieved_chunks)}")
        print(f"    Image Types      : {list(retrieved_image_types)}")
        print(f"    Groundedness     : {eval_metrics.get('groundedness')}/5")
        print(f"    Completeness     : {eval_metrics.get('completeness')}/5")
        print(f"    Correctness      : {eval_metrics.get('correctness')}/5")
        print("-" * 60)

    # 4. Save results to report
    report_data = {
        "report_metadata": {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_categories_tested": len(test_queries),
            "database_checked": os.path.basename(db_path),
            "evaluation_engine": "Gemini-3.1-Flash-Lite (LLM-as-a-judge)"
        },
        "retrieval_validation": validation_results
    }

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Validation report successfully saved to: {report_path}")
    except Exception as save_err:
        print(f"[ERROR] Failed to save report JSON: {save_err}")

    # Print overall quality averages
    avg_groundedness = sum(r["evaluation"].get("groundedness", 0) for r in validation_results) / len(validation_results)
    avg_completeness = sum(r["evaluation"].get("completeness", 0) for r in validation_results) / len(validation_results)
    avg_correctness = sum(r["evaluation"].get("correctness", 0) for r in validation_results) / len(validation_results)

    print("\n" + "=" * 60)
    print("                 RETRIEVAL SUITE QUALITY CARD")
    print("=" * 60)
    print(f"Average Groundedness Score : {avg_groundedness:.2f} / 5.0")
    print(f"Average Completeness Score : {avg_completeness:.2f} / 5.0")
    print(f"Average Correctness Score  : {avg_correctness:.2f} / 5.0")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
