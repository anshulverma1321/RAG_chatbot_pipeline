import os
from dotenv import load_dotenv
load_dotenv()

from RAG.query_engine import get_query_embedding
from RAG.vector_store import search_vectors
from RAG.db import get_sibling_chunks

DB_PATH = "RAG/data/rag_tool.db"
VECTOR_DB_PATH = "RAG/data/qdrant"
REPORT_PATH = "context_report.txt"

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    def print_retrieved_context(query):
        f.write(f"\n=======================================\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"=======================================\n")
        query_vector = get_query_embedding(query)
        search_results = search_vectors(VECTOR_DB_PATH, query_vector, None, 5)
        
        pages_to_load = {}
        for hit in search_results:
            payload = hit.get('payload')
            if not payload or 'document_id' not in payload or 'page_number' not in payload:
                continue
            doc_id = payload['document_id']
            page_num = payload['page_number']
            filename = payload.get('filename', f"Doc-{doc_id}")
            pages_to_load[(doc_id, page_num)] = filename
            
        f.write(f"Retrieved {len(pages_to_load)} unique pages:\n")
        for (doc_id, page_num), filename in pages_to_load.items():
            f.write(f"- {filename} (Page {page_num})\n")
            
        for (doc_id, page_num), filename in pages_to_load.items():
            f.write(f"\n--- Context block: {filename} (Page {page_num}) ---\n")
            page_chunks = get_sibling_chunks(DB_PATH, doc_id, page_num)
            page_content_parts = []
            for chunk in page_chunks:
                page_content_parts.append(chunk['content'])
            f.write("\n".join(page_content_parts) + "\n")

    print_retrieved_context("what is neural network?")
    print_retrieved_context("what is input space?")

print("Report written to", REPORT_PATH)
