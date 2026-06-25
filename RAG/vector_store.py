import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from concurrent.futures import ThreadPoolExecutor
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from RAG.exceptions import VectorStoreError, EmbeddingError

class PatchedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(self.embed_query, texts))

COLLECTION_NAME = "document_chunks"
VECTOR_DIMENSION = 3072  # gemini-embedding-2 output dimension

def get_qdrant_client(storage_path: str) -> QdrantClient:
    """Gets a local-directory client for Qdrant."""
    try:
        os.makedirs(storage_path, exist_ok=True)
        return QdrantClient(path=storage_path)
    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "get_qdrant_client", type(e).__name__, str(e))
        raise VectorStoreError("Vector database temporarily unavailable.") from e

def init_vector_store(storage_path: str):
    """Initializes the Qdrant local collection if it does not exist."""
    client = None
    try:
        client = get_qdrant_client(storage_path)
        
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
            )
    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "init_vector_store", type(e).__name__, str(e))
        if isinstance(e, VectorStoreError):
            raise
        raise VectorStoreError("Vector database temporarily unavailable.") from e
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

def get_vector_store(storage_path: str) -> QdrantVectorStore:
    """Gets a LangChain QdrantVectorStore client initialized with local client and PatchedGoogleGenerativeAIEmbeddings."""
    try:
        client = get_qdrant_client(storage_path)
        embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "get_vector_store", type(e).__name__, str(e))
        if isinstance(e, VectorStoreError):
            raise
        raise VectorStoreError("Vector database temporarily unavailable.") from e

def upsert_chunks(storage_path: str, chunks: List[Any]):
    """
    Upserts chunk texts to the local Qdrant collection using LangChain.
    Embeddings are automatically computed using GoogleGenerativeAIEmbeddings.
    Chunks are uploaded in small batches with exponential backoff to handle
    429 RESOURCE_EXHAUSTED and 503 UNAVAILABLE errors from the Gemini API.
    """
    if not chunks:
        return
        
    vector_store = None
    try:
        vector_store = get_vector_store(storage_path)
        
        documents = []
        ids = []
        for chunk in chunks:
            # Handle both dict and Pydantic model
            if hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump()
            elif hasattr(chunk, "dict"):
                chunk_dict = chunk.dict()
            else:
                chunk_dict = dict(chunk)

            # Validate ID is a valid UUID
            uid = chunk_dict.get('chunk_id') or chunk_dict.get('id')
            if not uid:
                uid = str(uuid.uuid4())
            try:
                uuid.UUID(uid)
            except ValueError:
                uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, uid))
                
            # Base metadata for backward compatibility
            meta = {
                "document_id": int(chunk_dict['document_id']),
                "page_number": int(chunk_dict['page_number']),
                "chunk_type": chunk_dict.get('chunk_type') or chunk_dict.get('asset_type'),
                "filename": chunk_dict.get('filename') or chunk_dict.get('document_name', '')
            }
            # Map extra legacy metadata fields if present
            for k in ["table_type", "classification_confidence", "classification_reason", 
                      "image_type", "confidence", "reason", "Image Type", "Confidence", "Reason"]:
                if k in chunk_dict:
                    meta[k] = chunk_dict[k]

            is_normalized = "embedding_text" in chunk_dict
            if is_normalized:
                # Add new metadata fields for the Qdrant payload
                meta["chunk_id"] = uid
                meta["document_name"] = chunk_dict.get("document_name")
                meta["asset_type"] = chunk_dict.get("asset_type")
                meta["classification_type"] = chunk_dict.get("classification_type")
                meta["extractor_used"] = chunk_dict.get("extractor_used")
                meta["rich_text_representation"] = chunk_dict.get("rich_text_representation")
                meta["metadata"] = chunk_dict.get("metadata") or {}
                
                # We embed ONLY chunk.embedding_text
                page_content = chunk_dict["embedding_text"]
            else:
                page_content = chunk_dict['content']

            doc = Document(
                page_content=page_content,
                metadata=meta
            )
            documents.append(doc)
            ids.append(uid)

        # --- Batched upsert with exponential backoff ---
        # Embedding large documents all at once triggers 429 quota errors.
        # We split into small batches and retry each batch with backoff.
        import time as _time
        import logging as _logging
        _vs_logger = _logging.getLogger(__name__)

        BATCH_SIZE = 5           # documents per embedding batch
        MAX_RETRIES = 5
        BASE_DELAY = 8.0         # seconds before first retry

        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, batch_start in enumerate(range(0, len(documents), BATCH_SIZE)):
            batch_docs = documents[batch_start: batch_start + BATCH_SIZE]
            batch_ids  = ids[batch_start: batch_start + BATCH_SIZE]

            _vs_logger.info(
                "QDRANT upsert batch %d/%d | chunks=%d",
                batch_idx + 1, total_batches, len(batch_docs),
            )

            for attempt in range(MAX_RETRIES):
                try:
                    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                    _vs_logger.info(
                        "QDRANT batch %d/%d success | attempt=%d",
                        batch_idx + 1, total_batches, attempt + 1,
                    )
                    break  # success — move to next batch
                except Exception as batch_exc:
                    err_str = str(batch_exc).lower()
                    is_rate_limit = any(k in err_str for k in ["429", "quota", "resource_exhausted", "rate"])
                    is_transient  = any(k in err_str for k in ["503", "unavailable", "timeout", "connect"])

                    if (is_rate_limit or is_transient) and attempt < MAX_RETRIES - 1:
                        delay = BASE_DELAY * (2 ** attempt)  # 8s, 16s, 32s, 64s, 128s
                        from RAG.logger import log_error
                        log_error(
                            "RAG.vector_store", "upsert_chunks_batch",
                            type(batch_exc).__name__,
                            f"Batch {batch_idx+1}/{total_batches} attempt {attempt+1}/{MAX_RETRIES} failed. "
                            f"Retrying in {delay:.0f}s..."
                        )
                        _vs_logger.warning(
                            "QDRANT batch upsert retry | batch=%d/%d | attempt=%d/%d | delay=%.0fs | error=%s",
                            batch_idx + 1, total_batches, attempt + 1, MAX_RETRIES, delay, batch_exc,
                        )
                        _time.sleep(delay)
                    else:
                        # Non-retryable or exhausted retries — propagate
                        raise

    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "upsert_chunks", type(e).__name__, str(e))
        err_msg = str(e).lower()
        if any(k in err_msg for k in ["api", "key", "credential", "quota", "limit", "network", "connect", "dns"]):
            raise EmbeddingError("Embedding generation failed. Please check API connectivity.") from e
        else:
            raise VectorStoreError("Vector database temporarily unavailable.") from e
    finally:
        if vector_store and hasattr(vector_store, 'client') and vector_store.client:
            try:
                vector_store.client.close()
            except Exception:
                pass

def search_vectors(
    storage_path: str, 
    query_vector: List[float], 
    document_ids: Optional[List[int]] = None, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Searches the vector store using LangChain's QdrantVectorStore and a pre-computed query vector.
    Applies metadata filtering using Qdrant filters on nested metadata fields.
    """
    vector_store = None
    try:
        vector_store = get_vector_store(storage_path)
        
        query_filter = None
        if document_ids:
            # Match any of the document IDs in the list (metadata is nested under "metadata" key in LangChain payloads)
            conditions = [
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=doc_id)
                ) for doc_id in document_ids
              ]
            query_filter = Filter(should=conditions)
            
        search_results = vector_store.similarity_search_with_score_by_vector(
            embedding=query_vector,
            k=top_k,
            filter=query_filter
        )
        
        results = []
        for doc, score in search_results:
            chunk_id = doc.id or doc.metadata.get("_id") or doc.metadata.get("chunk_id")
            results.append({
                "id": chunk_id,
                "score": score,
                "payload": doc.metadata,
                "content": doc.page_content
            })
        return results
    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "search_vectors", type(e).__name__, str(e))
        if isinstance(e, VectorStoreError):
            raise
        raise VectorStoreError("Vector database temporarily unavailable.") from e
    finally:
        if vector_store and hasattr(vector_store, 'client') and vector_store.client:
            try:
                vector_store.client.close()
            except Exception:
                pass

def delete_vectors_by_doc(storage_path: str, document_id: int):
    """Deletes all vectors belonging to a specific document ID."""
    client = None
    try:
        client = get_qdrant_client(storage_path)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
    except Exception as e:
        from RAG.logger import log_error
        log_error("RAG.vector_store", "delete_vectors_by_doc", type(e).__name__, str(e))
        if isinstance(e, VectorStoreError):
            raise
        raise VectorStoreError("Vector database temporarily unavailable.") from e
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
