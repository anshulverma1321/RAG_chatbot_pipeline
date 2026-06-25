import os
import hashlib
import uuid
import logging
import base64
import pdfplumber
from pypdf import PdfReader
from typing import List, Dict, Any, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from RAG.db import add_document, add_chunks, get_document_by_hash, delete_document, update_document_active_timestamp
from RAG.vector_store import upsert_chunks, PatchedGoogleGenerativeAIEmbeddings
from RAG.exceptions import (
    DocumentNotFoundError,
    InvalidFileTypeError,
    DuplicateDocumentError,
    CorruptedPDFError,
    IngestionError,
    RAGError
)

import tempfile
import json

logger = logging.getLogger(__name__)

def run_visual_understanding_logic_on_bytes(image_bytes: bytes, mime_type: str) -> dict:
    """Helper to write image bytes to a temp file and call run_visual_understanding_logic."""
    from RAG.services.image_intelligence import run_visual_understanding_logic
    ext = ".png"
    if mime_type == "image/jpeg":
        ext = ".jpg"
    elif mime_type == "image/webp":
        ext = ".webp"
        
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(image_bytes)
        return run_visual_understanding_logic(temp_path, ext)
    finally:
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_path}: {e}")


def resolve_content(result: Any) -> str:
    """Resolves the content of the image chunk prioritizing:
    rich_text_representation -> combined_understanding -> description -> serialized JSON"""
    if not result:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "rich_text_representation" in result and result["rich_text_representation"]:
            return result["rich_text_representation"]
        if "combined_understanding" in result and result["combined_understanding"]:
            return result["combined_understanding"]
        if "description" in result and result["description"]:
            return result["description"]
        if "vision_summary" in result and result["vision_summary"]:
            return result["vision_summary"]
        clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
        return json.dumps(clean_result, ensure_ascii=False)
    return str(result)


def get_file_hash(file_path: str) -> str:
    """Computes the SHA-256 hash of a file for duplicate check."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def table_to_markdown(table: List[List[Any]]) -> str:
    """Formats a raw table from pdfplumber as a clean Markdown table."""
    if not table or not table[0]:
        return ""
    # Clean up cells to convert None to empty strings and strip extra spaces
    cleaned = [[str(cell or "").strip() for cell in row] for row in table]
    headers = cleaned[0]
    num_cols = len(headers)
    rows = cleaned[1:]
    
    # Form markdown structure
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * num_cols) + " |\n"
    for row in rows:
        # Pad short rows so the Markdown table stays well-formed
        padded = row + [""] * (num_cols - len(row))
        markdown += "| " + " | ".join(padded[:num_cols]) + " |\n"
    return markdown

def describe_image_with_gemini(image_bytes: bytes, mime_type: str) -> str:
    """Calls Gemini-3.1-Flash-Lite VLM to generate a brief summary of a visual element."""
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Describe this chart, diagram, or image concisely in under 3 sentences for search indexing. "
            "Focus on: 1. Chart/Image type. 2. Main subject/variables. 3. Key trend or takeaway. "
            "Do not write conversational or introductory text (like 'Here is a description...')."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        )
        
        response = chat.invoke([message])
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
        return content.strip() if content else ""
    except Exception as e:
        logger.error("Error describing image with Gemini: %s", e)
        return "[Visual element failed to process]"

def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Splits plain text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    chunks = []
    if not text or len(text.strip()) == 0:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break on a space rather than cutting a word
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (chunk_size // 2):
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
        if start >= len(text) or chunk_size - overlap <= 0:
            break
            
    return [c for c in chunks if len(c) > 0]

def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Generates batch embeddings using PatchedGoogleGenerativeAIEmbeddings."""
    if not texts:
        return []
    try:
        embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return embeddings.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings with Gemini: {e}")

def process_pdf(file_path: str, db_path: str, vector_db_path: str) -> Tuple[int, str]:
    """
    Ingests a PDF:
    - Extracts text, tables, and describes images page-by-page.
    - Saves metadata to SQLite.
    - Saves vectors to Qdrant.
    Returns (document_id, message).
    """
    import time
    from RAG.logger import log_ingestion_start, log_ingestion_success, log_error, log_performance
    from RAG.services.image_classifier import classify_image
    from RAG.services.text_knowledge_extractor import extract_text_knowledge
    from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
    from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge


    filename = os.path.basename(file_path)

    # 1. File existence validation
    if not os.path.exists(file_path):
        raise DocumentNotFoundError(
            f"File:\n{filename}\n\nError:\nFile not found.\n\nPlease verify the filename or path."
        )

    # 2. File type validation
    if not file_path.lower().endswith(".pdf"):
        raise InvalidFileTypeError("Only PDF files are supported.")

    try:
        file_hash = get_file_hash(file_path)
    except Exception as e:
        log_error("ingestion.py", "get_file_hash", type(e).__name__, str(e))
        raise IngestionError(f"Failed to read file: {e}") from e
    
    # 3. Duplicate check
    try:
        existing_doc = get_document_by_hash(db_path, file_hash)
    except Exception as e:
        # SQLite error wrapped in DatabaseError
        raise
        
    if existing_doc:
        raise DuplicateDocumentError("Document already exists in the knowledge base.")
    
    start_total = time.time()
    log_ingestion_start(filename)
    
    doc_id = None
    try:
        # 4. Open and parse PDF
        try:
            pypdf_reader = PdfReader(file_path)
            total_pages = len(pypdf_reader.pages)
        except Exception as e:
            log_error("ingestion.py", "open_pdf", type(e).__name__, str(e))
            raise CorruptedPDFError("The PDF could not be processed because it appears to be corrupted.") from e
        
        # Add document to SQLite metadata
        doc_id = add_document(db_path, filename, file_hash, total_pages)
        
        start_processing = time.time()
        
        from RAG.document_orchestrator import orchestrate_pdf
        normalized_chunks = orchestrate_pdf(file_path)
        
        pdf_proc_time = time.time() - start_processing
        
        if not normalized_chunks:
            raise IngestionError(f"Uploaded '{filename}', but no content (text, tables, or images) was successfully extracted.")
            
        start_db_write = time.time()
        
        # Prepare list for SQLite metadata storage
        chunks_to_insert = []
        for idx, c in enumerate(normalized_chunks):
            # Include structured_knowledge inside SQLite metadata column to allow retrieval
            db_metadata = dict(c.metadata) if isinstance(c.metadata, dict) else {}
            db_metadata["structured_knowledge"] = c.structured_knowledge
            
            chunks_to_insert.append({
                "id": c.chunk_id,
                "document_id": doc_id,
                "page_number": c.page_number,
                "chunk_type": c.asset_type,
                "content": c.rich_text_representation,
                "sibling_order": idx,
                "asset_type": c.asset_type,
                "classification_type": c.classification_type,
                "extractor_used": c.extractor_used,
                "document_name": c.document_name,
                "metadata": db_metadata
            })
            
        # Write to SQLite
        add_chunks(db_path, chunks_to_insert)
        logger.info("SQLite insertion success | total_chunks=%d", len(chunks_to_insert))
        
        # Prepare list for Qdrant (which will generate embeddings on embedding_text)
        qdrant_chunks = []
        for c in normalized_chunks:
            chunk_dict = c.model_dump() if hasattr(c, "model_dump") else c.dict()
            chunk_dict["document_id"] = doc_id
            qdrant_chunks.append(chunk_dict)
            
        # Write to Qdrant Vector DB
        upsert_chunks(vector_db_path, qdrant_chunks)
        logger.info("Qdrant insertion success | total_chunks=%d", len(qdrant_chunks))
        
        db_write_time = time.time() - start_db_write
        total_time = time.time() - start_total
        
        images_count = sum(1 for c in normalized_chunks if c.asset_type == "image")
        tables_count = sum(1 for c in normalized_chunks if c.asset_type == "table")
        
        log_ingestion_success(filename, total_pages, len(normalized_chunks), images_count, tables_count)
        log_performance({
            "pdf_processing": pdf_proc_time,
            "embedding_generation_and_indexing": db_write_time,
            "total_ingestion": total_time
        })
        
        # Update last_active_at timestamp for newly ingested PDF
        try:
            update_document_active_timestamp(db_path, doc_id)
        except Exception as exc:
            logger.error("Failed to update active document timestamp for PDF: %s", exc)
        
    except RAGError:
        # Re-raise clean custom exceptions (e.g. from SQLite or Qdrant/Embedding)
        # Rollback SQLite record if it was partially created
        if doc_id is not None:
            logger.error("Ingestion failed, rolling back document record ID: %s", doc_id)
            try:
                delete_document(db_path, doc_id)
            except Exception as rollback_err:
                logger.error("Rollback also failed: %s", rollback_err)
        raise
    except Exception as e:
        log_error("ingestion.py", "process_pdf", type(e).__name__, str(e))
        # Rollback SQLite record
        if doc_id is not None:
            logger.error("Ingestion failed, rolling back document record ID: %s", doc_id)
            try:
                delete_document(db_path, doc_id)
            except Exception as rollback_err:
                logger.error("Rollback also failed: %s", rollback_err)
        raise IngestionError(f"Failed to ingest {filename}: {e}") from e
    
    return doc_id, f"Successfully processed '{filename}' ({total_pages} pages, {len(chunks_to_insert)} chunks created)."
