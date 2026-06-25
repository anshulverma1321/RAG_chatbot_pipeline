import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

from RAG.vector_store import search_vectors
from RAG.query_engine import get_query_embedding

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    query = "What is image segmentation?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"Running retrieval test for query:\n\"{query}\"")
    
    # Resolve databases
    vector_db_path = os.path.join(base_dir, "RAG", "data", "qdrant")
    
    try:
        # Generate embedding
        print("Generating query embedding...")
        query_vector = get_query_embedding(query)
        
        # Search vector store directly
        print("Searching Qdrant vector store...")
        results = search_vectors(vector_db_path, query_vector, top_k=5)
        
        print("\nTop Results:\n")
        
        out_lines = []
        out_lines.append(f"Query: {query}\n\nTop Results:\n")
        
        for idx, r in enumerate(results, 1):
            payload = r.get('payload', {})
            filename = payload.get('filename', f"Doc-{payload.get('document_id')}")
            page_num = payload.get('page_number', 'unknown')
            score = r.get('score', 0.0)
            content = r.get('content', '')
            preview = content[:200].replace('\n', ' ') + ("..." if len(content) > 200 else "")
            
            result_block = (
                f"{idx}.\n"
                f"File:\n"
                f"{filename}\n\n"
                f"Page:\n"
                f"{page_num}\n\n"
                f"Score:\n"
                f"{score:.2f}\n\n"
                f"Preview:\n"
                f"{preview}\n"
                f"{'-'*40}\n"
            )
            safe_print(result_block)
            out_lines.append(result_block)
            
        # Save output
        out_dir = os.path.join(base_dir, "outputs", "retrieval")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "retrieval_results.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            
        print(f"Retrieval results saved to: {out_file}")
    except Exception as e:
        print(f"Error performing retrieval search: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
