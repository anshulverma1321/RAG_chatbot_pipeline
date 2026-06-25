import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

from RAG.vector_store import PatchedGoogleGenerativeAIEmbeddings

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    text = "Neural network architectures for document analysis, including table parsing and visual elements representation."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        
    print(f"Generating embedding for text:\n\"{text}\"")
    
    try:
        embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        vector = embeddings.embed_query(text)
        
        print("\nEmbedding Generated")
        print("\nDimensions:")
        print(len(vector))
        
        # Verify
        if len(vector) != 3072:
            print(f"Warning: Dimensions expected 3072, but got {len(vector)}")
            
        has_null = any(x is None for x in vector)
        print(f"\nNo null vectors: {not has_null}")
        
        preview = vector[:5]
        preview_str = ", ".join(f"{x:.3f}" for x in preview)
        safe_print(f"\nVector Preview:\n[{preview_str}, ...]")
        
        # Save output
        out_dir = os.path.join(base_dir, "outputs", "embeddings")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "embedding_result.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"Text:\n{text}\n\nDimensions: {len(vector)}\nVector:\n{vector}\n")
            
        print(f"\nEmbedding result saved to: {out_file}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
