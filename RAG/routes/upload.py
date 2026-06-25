"""
RAG/routes/upload.py

Phase 8 — Async Upload Endpoint
=================================
POST /upload  → validates file, registers an ingestion job, and returns a
               job_id instantly (HTTP 202 Accepted).  The heavy extraction
               pipeline runs in a background thread.

GET  /jobs/{job_id} → poll job status (pending / running / success / failed).

Workflow (background thread):
  Save file → duplicate check → PDF / Image / Spreadsheet pipeline
  → normalize → SQLite → Qdrant → mark job 'success'

Internal helpers (_build_processing_strategy, ClassificationDetail, RoutingResult)
are kept here so that process.py can still import them.
"""

import io as _io
import os
import hashlib
import time
import logging
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from RAG.content_router import route_content, SUPPORTED_EXTENSIONS, RoutingMetadata
from RAG.knowledge_normalizer import NormalizedKnowledgeChunk, normalize_knowledge_chunk
from RAG.exceptions import DuplicateDocumentError
from RAG.db import (
    get_document_by_hash,
    add_document,
    create_job,
    update_job,
    get_job,
    get_job_by_hash,
    update_document_active_timestamp,
)
from RAG.vector_store import upsert_chunks
from RAG.document_orchestrator import orchestrate_pdf
from pypdf import PdfReader

# Imports for structured logging and context propagation
from RAG.logger import (
    request_id_var,
    document_id_var,
    file_name_var,
    execution_stage_var,
    performance_logger,
    init_performance_timings,
    record_performance_timing,
    get_performance_timings
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths — resolved relative to the RAG package root
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RAG/
_DATA_DIR = os.path.join(_BASE_DIR, "data")
DB_PATH = os.path.join(_DATA_DIR, "rag_tool.db")
VECTOR_DB_PATH = os.path.join(_DATA_DIR, "qdrant")
_TEMP_DIR = os.path.join(_DATA_DIR, "temp")
os.makedirs(_TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Internal helpers (kept here so process.py can import them)
# ---------------------------------------------------------------------------

def _build_processing_strategy(primary_category: str, classification: Optional[Dict[str, Any]]) -> str:
    """
    Derives a compact processing strategy label from routing metadata.

    Rules
    -----
    - document                          → "pdf_document"
    - image   + CHART classification    → "image_chart"
    - image   + DIAGRAM                 → "image_diagram"
    - image   + TEXT_IMAGE              → "image_text"
    - image   + MIXED                   → "image_mixed"
    - image   + NATURAL_IMAGE           → "image_natural"
    - image   (no dynamic class)        → "image_unknown"
    - spreadsheet + TABLE_FINANCIAL     → "table_financial"
    - spreadsheet + TABLE_STATISTICAL   → "table_statistical"
    - spreadsheet + TABLE_TIMESERIES    → "table_timeseries"
    - spreadsheet + TABLE_COMPARISON    → "table_comparison"
    - spreadsheet + TABLE_SIMPLE        → "table_simple"
    - spreadsheet (no dynamic class)    → "table_unknown"
    - unknown                           → "unsupported"
    """
    if primary_category == "document":
        return "pdf_document"

    if primary_category == "image":
        if classification:
            raw_type = classification.get("type", "").upper()
            label_map = {
                "CHART": "image_chart",
                "DIAGRAM": "image_diagram",
                "TEXT_IMAGE": "image_text",
                "MIXED": "image_mixed",
                "NATURAL_IMAGE": "image_natural",
            }
            return label_map.get(raw_type, "image_unknown")
        return "image_unknown"

    if primary_category == "spreadsheet":
        if classification:
            raw_type = classification.get("type", "").upper()
            label_map = {
                "TABLE_FINANCIAL": "table_financial",
                "TABLE_STATISTICAL": "table_statistical",
                "TABLE_TIMESERIES": "table_timeseries",
                "TABLE_COMPARISON": "table_comparison",
                "TABLE_SIMPLE": "table_simple",
                "TABLE_UNKNOWN": "table_unknown",
            }
            return label_map.get(raw_type, "table_unknown")
        return "table_unknown"

    return "unsupported"


def _compute_hash(data: bytes) -> str:
    """Computes SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Response schemas (kept for backward compatibility with process.py imports)
# ---------------------------------------------------------------------------

class ClassificationDetail(BaseModel):
    """Optional dynamic classifier output when content analysis was performed."""
    type: str = Field(..., description="Classifier label (e.g. CHART, TABLE_FINANCIAL).")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    reason: str = Field(..., description="Human-readable classification rationale.")


class RoutingResult(BaseModel):
    """Full routing decision returned for an uploaded file."""
    file_name: str = Field(..., description="Original filename of the upload.")
    file_extension: str = Field(..., description="Normalized lowercase extension without leading dot.")
    mime_type: str = Field(..., description="Resolved MIME type.")
    primary_category: str = Field(
        ...,
        description="High-level content category: 'document', 'image', 'spreadsheet', or 'unknown'.",
    )
    is_supported: bool = Field(..., description="True if the format is supported.")
    suggested_route: str = Field(
        ...,
        description="Pipeline routing name: 'pdf_pipeline', 'image_pipeline', 'table_pipeline', or 'unknown_pipeline'.",
    )
    suggested_extractor: str = Field(
        ...,
        description="Name of the recommended extractor function for downstream processing.",
    )
    processing_strategy: str = Field(
        ...,
        description=(
            "Compact strategy label combining category and sub-type. "
            "Examples: 'pdf_document', 'image_chart', 'table_financial', 'image_unknown'."
        ),
    )
    classification: Optional[ClassificationDetail] = Field(
        None,
        description="Dynamic classifier output when content analysis was performed (images & spreadsheets with bytes).",
    )


# ---------------------------------------------------------------------------
# Phase 8 — Async Upload Response Schema
# ---------------------------------------------------------------------------

class UploadAcceptedResponse(BaseModel):
    """Returned immediately (HTTP 202) after the upload is accepted."""
    status: str = Field(..., description="'accepted' or 'already_ingested'.")
    job_id: str = Field(..., description="Use this ID to poll GET /jobs/{job_id}.")
    file_name: str = Field(..., description="Original filename of the upload.")
    file_type: str = Field(..., description="Normalized file extension (e.g. 'pdf', 'png', 'csv').")
    message: str = Field(..., description="Human-readable status message.")
    document_id: Optional[int] = Field(None, description="The ID of the document if already ingested.")


class JobStatusResponse(BaseModel):
    """Returned by GET /jobs/{job_id}."""
    job_id: str
    status: str = Field(..., description="pending | running | success | failed | already_ingested")
    file_name: str
    document_id: Optional[int] = None
    pages_processed: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Backward-compat alias used by old code that imported UploadIngestionResponse
# ---------------------------------------------------------------------------

class UploadIngestionResponse(BaseModel):
    """Legacy response schema kept for backward compatibility."""
    status: str
    document_id: int
    file_name: str
    file_type: str
    pages_processed: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    ingested: bool
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal ingestion helpers
# ---------------------------------------------------------------------------

def _ingest_image_to_chunks(
    file_bytes: bytes,
    safe_name: str,
    routing_meta: RoutingMetadata,
) -> List[NormalizedKnowledgeChunk]:
    """
    Routes an image through the appropriate intelligence extractor and returns
    a list containing one NormalizedKnowledgeChunk.
    """
    stage_token = execution_stage_var.set("image_pipeline")
    vlm_type = ""
    if routing_meta.classification:
        vlm_type = routing_meta.classification.get("type", "").upper()

    extraction_result: Dict[str, Any] = {}
    selected_extractor = ""

    logger.info("IMAGE PIPELINE classifier selected | type=%s", vlm_type or "UNKNOWN")
    try:
        if vlm_type == "TEXT_IMAGE":
            from RAG.services.text_knowledge_extractor import extract_text_knowledge
            selected_extractor = "extract_text_knowledge"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_text_knowledge(file_bytes, routing_meta.mime_type)
        elif vlm_type == "CHART":
            from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
            selected_extractor = "extract_chart_knowledge"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_chart_knowledge(file_bytes, routing_meta.mime_type)
        elif vlm_type == "DIAGRAM":
            from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
            selected_extractor = "extract_diagram_knowledge"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_diagram_knowledge(file_bytes, routing_meta.mime_type)
        elif vlm_type == "MIXED":
            from RAG.ingestion import run_visual_understanding_logic_on_bytes
            selected_extractor = "run_visual_understanding_logic"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = run_visual_understanding_logic_on_bytes(file_bytes, routing_meta.mime_type)
        elif vlm_type == "NATURAL_IMAGE":
            from RAG.services.image_intelligence import extract_natural_image_knowledge
            selected_extractor = "extract_natural_image_knowledge"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_natural_image_knowledge(file_bytes, routing_meta.mime_type)
        else:
            from RAG.ingestion import describe_image_with_gemini
            selected_extractor = "describe_image_with_gemini"
            logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
            desc = describe_image_with_gemini(file_bytes, routing_meta.mime_type)
            extraction_result = {
                "summary": desc,
                "rich_text_representation": f"# Image Description\n\n## Gemini Vision Summary:\n{desc}\n",
            }
        logger.info("IMAGE PIPELINE extraction completed | extractor=%s", selected_extractor)
    except Exception as exc:
        logger.exception("Image extractor %s failed for %s: %s", selected_extractor, safe_name, exc)
        # Graceful fallback
        try:
            from RAG.ingestion import describe_image_with_gemini
            selected_extractor = "describe_image_with_gemini (fallback)"
            desc = describe_image_with_gemini(file_bytes, routing_meta.mime_type)
            extraction_result = {
                "summary": desc,
                "rich_text_representation": f"# Image Description (Fallback)\n\n## Gemini Vision Summary:\n{desc}\n",
            }
        except Exception as fallback_exc:
            logger.exception("Fallback image descriptor also failed: %s", fallback_exc)
            extraction_result = {
                "summary": "Failed to process visual content.",
                "rich_text_representation": "# Visual Element Description Failed\n",
            }
    finally:
        execution_stage_var.reset(stage_token)

    rich_text = extraction_result.pop("rich_text_representation", "")
    chunk = normalize_knowledge_chunk(
        document_name=safe_name,
        page_number=1,
        asset_type="image",
        classification_type=vlm_type or "UNKNOWN",
        extractor_used=selected_extractor,
        knowledge_object=extraction_result,
        rich_text_representation=rich_text,
    )
    return [chunk]


def _ingest_spreadsheet_to_chunks(
    file_bytes: bytes,
    safe_name: str,
    routing_meta: RoutingMetadata,
) -> List[NormalizedKnowledgeChunk]:
    """
    Parses a spreadsheet to Markdown, routes through the appropriate table extractor
    and returns a list containing one NormalizedKnowledgeChunk.
    """
    stage_token = execution_stage_var.set("table_pipeline")
    from RAG.services.table_intelligence import (
        extract_simple_table,
        extract_comparison_table,
        extract_financial_table,
        extract_statistical_table,
        extract_timeseries_table,
        extract_unknown_table,
    )

    # Write bytes to a temp file so parse_table_file_to_markdown can read it
    ext = routing_meta.file_extension
    suffix = f".{ext}" if ext else ""
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=_TEMP_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)

        from RAG.routes.validation import parse_table_file_to_markdown
        table_markdown = parse_table_file_to_markdown(temp_path, safe_name)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    tbl_type = ""
    if routing_meta.classification:
        tbl_type = routing_meta.classification.get("type", "").upper()

    extraction_result: Dict[str, Any] = {}
    selected_extractor = ""

    logger.info("TABLE PIPELINE classifier selected | type=%s", tbl_type or "TABLE_UNKNOWN")
    try:
        if tbl_type == "TABLE_SIMPLE":
            selected_extractor = "extract_simple_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_simple_table(table_markdown)
        elif tbl_type == "TABLE_COMPARISON":
            selected_extractor = "extract_comparison_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_comparison_table(table_markdown)
        elif tbl_type == "TABLE_FINANCIAL":
            selected_extractor = "extract_financial_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_financial_table(table_markdown)
        elif tbl_type == "TABLE_STATISTICAL":
            selected_extractor = "extract_statistical_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_statistical_table(table_markdown)
        elif tbl_type == "TABLE_TIMESERIES":
            selected_extractor = "extract_timeseries_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_timeseries_table(table_markdown)
        else:
            selected_extractor = "extract_unknown_table"
            logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
            extraction_result = extract_unknown_table(table_markdown)
        logger.info("TABLE PIPELINE extraction completed | extractor=%s", selected_extractor)
    except Exception as exc:
        logger.exception("Table extractor %s failed for %s: %s", selected_extractor, safe_name, exc)
        try:
            selected_extractor = "extract_unknown_table (fallback)"
            extraction_result = extract_unknown_table(table_markdown)
        except Exception as fallback_exc:
            logger.exception("Fallback table extractor also failed: %s", fallback_exc)
            extraction_result = {
                "title": "Table",
                "summary": "Table extraction failed.",
                "rich_text_representation": table_markdown,
            }
    finally:
        execution_stage_var.reset(stage_token)

    rich_text = extraction_result.pop("rich_text_representation", "")
    chunk = normalize_knowledge_chunk(
        document_name=safe_name,
        page_number=1,
        asset_type="table",
        classification_type=tbl_type or "TABLE_UNKNOWN",
        extractor_used=selected_extractor,
        knowledge_object=extraction_result,
        rich_text_representation=rich_text,
    )
    return [chunk]


def _store_chunks(
    db_path: str,
    vector_db_path: str,
    doc_id: int,
    normalized_chunks: List[NormalizedKnowledgeChunk],
) -> None:
    """Writes normalized chunks to SQLite and Qdrant."""
    import json
    import time
    from RAG.db import add_chunks
    
    stage_token = execution_stage_var.set("ingestion")
    try:
        chunks_to_insert = []
        for idx, c in enumerate(normalized_chunks):
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
                "metadata": db_metadata,
            })

        logger.info("INGESTION SQLite insert | chunks_count=%d", len(chunks_to_insert))
        start_sqlite = time.time()
        add_chunks(db_path, chunks_to_insert)
        sqlite_time = time.time() - start_sqlite
        logger.info("SQLite insertion success | doc_id=%d | total_chunks=%d", doc_id, len(chunks_to_insert))

        qdrant_chunks = []
        for c in normalized_chunks:
            chunk_dict = c.model_dump() if hasattr(c, "model_dump") else c.dict()
            chunk_dict["document_id"] = doc_id
            qdrant_chunks.append(chunk_dict)

        logger.info("INGESTION Qdrant upsert | chunks_count=%d", len(qdrant_chunks))
        start_qdrant = time.time()
        upsert_chunks(vector_db_path, qdrant_chunks)
        qdrant_time = time.time() - start_qdrant
        logger.info("Qdrant insertion success | doc_id=%d | total_chunks=%d", doc_id, len(qdrant_chunks))
        
        # Accumulate ingestion time
        record_performance_timing("ingestion_time", sqlite_time + qdrant_time)
    finally:
        execution_stage_var.reset(stage_token)


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------

def _run_ingestion_job(
    job_id: str,
    file_bytes: bytes,
    safe_name: str,
    ext: str,
    file_hash: str,
    request_id: Optional[str] = None,
) -> None:
    """
    Runs the full ingestion pipeline in a background thread.
    Updates the ingestion_jobs row as it progresses.
    """
    # Propagate request context
    if request_id:
        request_id_var.set(request_id)
    file_name_var.set(safe_name)
    stage_token = execution_stage_var.set("upload")
    
    # Initialize performance timings
    init_performance_timings()
    start_total_job = time.time()

    logger.info("UPLOAD ingestion started | job_id=%s | file=%s", job_id, safe_name)
    logger.info("[JOB START] job_id=%s | file=%s", job_id, safe_name)

    reingest_msg = (
        f"\n[REINGEST START]\n"
        f"job_id={job_id}\n"
        f"file={safe_name}\n"
        f"ext={ext}\n"
    )
    logger.info(reingest_msg)
    print(reingest_msg)

    # Mark as running
    update_job(DB_PATH, job_id, status="running")

    doc_id: Optional[int] = None

    try:
        # --- Route content ---
        start_route = time.time()
        routing_meta: RoutingMetadata = route_content(
            file_name=safe_name,
            file_bytes=file_bytes,
            mime_type=None,
        )
        record_performance_timing("routing_time", time.time() - start_route)
        logger.info("UPLOAD routing result | category=%s", routing_meta.primary_category)

        normalized_chunks: List[NormalizedKnowledgeChunk] = []
        pages_processed = 1

        # Measure extraction/processing duration
        start_ext = time.time()

        if routing_meta.primary_category == "document":
            suffix = f".{routing_meta.file_extension}"
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=_TEMP_DIR)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(file_bytes)

                try:
                    reader = PdfReader(_io.BytesIO(file_bytes))
                    pages_processed = len(reader.pages)
                except Exception:
                    pages_processed = 1

                doc_id = add_document(DB_PATH, safe_name, file_hash, pages_processed)
                document_id_var.set(doc_id)
                # Stamp last_active_at immediately so this document becomes the active
                # document even while ingestion is still running in the background.
                # Without this, any query fired before the job succeeds will miss it.
                try:
                    update_document_active_timestamp(DB_PATH, doc_id)
                except Exception as _ts_err:
                    logger.warning("[REINGEST] Could not pre-stamp last_active_at for doc_id=%d: %s", doc_id, _ts_err)
                doc_created_msg = (
                    f"\n[REINGEST DOCUMENT CREATED]\n"
                    f"document_id={doc_id}\n"
                    f"filename={safe_name}\n"
                    f"pages={pages_processed}\n"
                    f"last_active_at=pre_stamped\n"
                )
                logger.info(doc_created_msg)
                print(doc_created_msg)
                normalized_chunks = orchestrate_pdf(temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        elif routing_meta.primary_category == "image":
            doc_id = add_document(DB_PATH, safe_name, file_hash, 1)
            document_id_var.set(doc_id)
            try:
                update_document_active_timestamp(DB_PATH, doc_id)
            except Exception as _ts_err:
                logger.warning("[REINGEST] Could not pre-stamp last_active_at for doc_id=%d: %s", doc_id, _ts_err)
            doc_created_msg = (
                f"\n[REINGEST DOCUMENT CREATED]\n"
                f"document_id={doc_id}\n"
                f"filename={safe_name}\n"
                f"pages=1\n"
                f"last_active_at=pre_stamped\n"
            )
            logger.info(doc_created_msg)
            print(doc_created_msg)
            normalized_chunks = _ingest_image_to_chunks(file_bytes, safe_name, routing_meta)

        elif routing_meta.primary_category == "spreadsheet":
            doc_id = add_document(DB_PATH, safe_name, file_hash, 1)
            document_id_var.set(doc_id)
            try:
                update_document_active_timestamp(DB_PATH, doc_id)
            except Exception as _ts_err:
                logger.warning("[REINGEST] Could not pre-stamp last_active_at for doc_id=%d: %s", doc_id, _ts_err)
            doc_created_msg = (
                f"\n[REINGEST DOCUMENT CREATED]\n"
                f"document_id={doc_id}\n"
                f"filename={safe_name}\n"
                f"pages=1\n"
                f"last_active_at=pre_stamped\n"
            )
            logger.info(doc_created_msg)
            print(doc_created_msg)
            normalized_chunks = _ingest_spreadsheet_to_chunks(file_bytes, safe_name, routing_meta)

        else:
            raise ValueError(f"Unsupported content category: '{routing_meta.primary_category}'")

        ext_duration = time.time() - start_ext
        
        # Calculate pure extraction time by subtracting dynamic classification time
        timings = get_performance_timings()
        if timings:
            net_ext = max(0.0, ext_duration - timings.get("classification_time", 0.0))
            record_performance_timing("extraction_time", net_ext)

        if not normalized_chunks:
            raise ValueError(f"No content could be extracted from '{safe_name}'.")

        # Store chunks
        text_chunks = sum(1 for c in normalized_chunks if c.asset_type == "text")
        table_chunks = sum(1 for c in normalized_chunks if c.asset_type == "table")
        image_chunks = sum(1 for c in normalized_chunks if c.asset_type == "image")
        total_chunks = len(normalized_chunks)

        chunks_msg = (
            f"\n[REINGEST CHUNKS READY]\n"
            f"document_id={doc_id}\n"
            f"filename={safe_name}\n"
            f"total_chunks={total_chunks}\n"
            f"text_chunks={text_chunks}\n"
            f"table_chunks={table_chunks}\n"
            f"image_chunks={image_chunks}\n"
            f"status=about_to_store\n"
        )
        logger.info(chunks_msg)
        print(chunks_msg)

        _store_chunks(DB_PATH, VECTOR_DB_PATH, doc_id, normalized_chunks)

        stored_msg = (
            f"\n[REINGEST CHUNKS INSERTED]\n"
            f"document_id={doc_id}\n"
            f"filename={safe_name}\n"
            f"total_chunks={total_chunks}\n"
            f"status=sqlite_and_qdrant_complete\n"
        )
        logger.info(stored_msg)
        print(stored_msg)

        qdrant_msg = (
            f"\n[REINGEST QDRANT UPSERT COMPLETE]\n"
            f"document_id={doc_id}\n"
            f"filename={safe_name}\n"
            f"chunks_upserted={total_chunks}\n"
        )
        logger.info(qdrant_msg)
        print(qdrant_msg)

        update_job(
            DB_PATH,
            job_id,
            status="success",
            document_id=doc_id,
            pages_processed=pages_processed,
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            image_chunks=image_chunks,
            total_chunks=total_chunks,
        )

        # Update last_active_at timestamp for newly ingested document
        try:
            update_document_active_timestamp(DB_PATH, doc_id)
            ts_msg = (
                f"\n[REINGEST ACTIVE TIMESTAMP UPDATED]\n"
                f"document_id={doc_id}\n"
                f"filename={safe_name}\n"
            )
            logger.info(ts_msg)
            print(ts_msg)
        except Exception as exc:
            logger.error("Failed to update active document timestamp on successful job: %s", exc)

        finished_msg = (
            f"\n[REINGEST FINISHED]\n"
            f"job_id={job_id}\n"
            f"document_id={doc_id}\n"
            f"filename={safe_name}\n"
            f"pages={pages_processed}\n"
            f"total_chunks={total_chunks}\n"
            f"status=success\n"
        )
        logger.info(finished_msg)
        print(finished_msg)

        logger.info(
            "[JOB SUCCESS] job_id=%s | file=%s | doc_id=%d | pages=%d | chunks=%d",
            job_id, safe_name, doc_id, pages_processed, total_chunks,
        )
        logger.info("UPLOAD ingestion completed")
        
        # Log final performance metrics
        final_timings = get_performance_timings()
        if final_timings:
            final_timings["total_time"] = time.time() - start_total_job
            performance_logger.info("UPLOAD timings", extra={"metrics": final_timings})

    except Exception as exc:
        logger.exception("[JOB FAILED] job_id=%s | file=%s | error=%s", job_id, safe_name, exc)

        # Conditionally rollback the document record.
        # Only delete if NO chunks were stored yet — this avoids destroying a
        # partially-ingested document that the stale-cleanup can recover later.
        # If chunks do exist (e.g. embedding failed mid-way) keep the record so
        # the user can at least query what was indexed before the failure.
        if doc_id is not None:
            try:
                from RAG.db import get_db_connection as _gdc
                _conn = _gdc(DB_PATH)
                _cur = _conn.cursor()
                _cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
                _chunk_cnt = _cur.fetchone()[0]
                _conn.close()
            except Exception:
                _chunk_cnt = 0

            if _chunk_cnt == 0:
                # No chunks stored — safe to delete the empty document record
                try:
                    from RAG.db import delete_document
                    delete_document(DB_PATH, doc_id)
                    logger.warning(
                        "[JOB ROLLBACK] Deleted empty doc_id=%d for job_id=%s (0 chunks)",
                        doc_id, job_id,
                    )
                except Exception as rollback_err:
                    logger.exception("[JOB ROLLBACK FAILED] doc_id=%d: %s", doc_id, rollback_err)
            else:
                # Chunks exist — keep the document, just mark the job failed
                logger.warning(
                    "[JOB PARTIAL] doc_id=%d kept with %d chunk(s) despite job failure — "
                    "stale-cleanup will recover on next upload | job_id=%s",
                    doc_id, _chunk_cnt, job_id,
                )

        update_job(
            DB_PATH,
            job_id,
            status="failed",
            error_message=str(exc)[:1000],
        )
    finally:
        execution_stage_var.reset(stage_token)


# ---------------------------------------------------------------------------
# POST /upload  — Async Ingestion Endpoint (Phase 8)
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload & Ingest Document (Async)",
    description=(
        "**Single Public Ingestion Endpoint**\n\n"
        "Accepts any supported file and immediately returns a `job_id` (HTTP 202). "
        "The heavy extraction pipeline runs in the background.\n\n"
        "**Poll status:** `GET /jobs/{job_id}`\n\n"
        "**Supported formats:** PDF · PNG · JPG · JPEG · CSV · XLSX · XLS\n\n"
        "**Pipeline (background):**\n"
        "1. Route content (AI classification)\n"
        "2. Extract knowledge (text, tables, images)\n"
        "3. Normalize chunks\n"
        "4. Generate embeddings\n"
        "5. Store into SQLite\n"
        "6. Store into Qdrant\n\n"
        "Uploading the same file twice returns `already_ingested` (HTTP 200) — not an error."
    ),
    tags=["Upload"],
)
async def universal_upload(
    file: UploadFile = File(..., description="The file to upload and ingest."),
):
    """
    Async ingestion: validate → duplicate-check → spawn background thread → return job_id.
    Poll GET /jobs/{job_id} to check progress.
    """
    file_name_var.set(file.filename)
    logger.info("UPLOAD upload started | filename=%s", file.filename)
    logger.info("[UPLOAD START] POST /upload received | filename=%s", file.filename)

    # --- 1. Filename validation ---
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided. Please supply a valid file.",
        )

    safe_name = os.path.basename(file.filename.strip())
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    # --- 2. Extension check ---
    ext = os.path.splitext(safe_name)[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS)).upper()}."
            ),
        )

    # --- 3. Read bytes ---
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file %s: %s", safe_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 50 MB upload limit.",
        )

    # --- 4. Gemini API key check ---
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured on the server.",
        )

    # --- 5. Duplicate check (hash-based) ---
    file_hash = _compute_hash(file_bytes)

    try:
        existing_doc = get_document_by_hash(DB_PATH, file_hash)
    except Exception as exc:
        logger.error("Duplicate check failed for %s: %s", safe_name, exc)
        existing_doc = None

    if existing_doc:
        try:
            from RAG.db import get_db_connection
            conn = get_db_connection(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (existing_doc["id"],))
            cnt = cursor.fetchone()[0]
            conn.close()
        except Exception:
            cnt = 0

        # [DUPLICATE CHECK]
        dup_msg = (
            f"\n[DUPLICATE CHECK]\n"
            f"document_id={existing_doc['id']}\n"
            f"file_name={safe_name}\n"
            f"chunk_count={cnt}\n"
        )
        logger.info(dup_msg)
        print(dup_msg)

        # Check if there is an active running/pending job for this file hash
        # A job that has been stuck in "running" for more than 15 minutes
        # is considered dead (e.g. the server restarted mid-job) and is NOT treated
        # as active — this allows the stale cleanup path to proceed.
        _STALE_JOB_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
        is_job_active = False
        try:
            import datetime as _dt
            existing_job = get_job_by_hash(DB_PATH, file_hash)
            if existing_job and existing_job["status"] in ("pending", "running"):
                # Check whether this job has been active recently
                updated_at_str = existing_job.get("updated_at") or existing_job.get("created_at")
                job_is_stale = False
                if updated_at_str:
                    try:
                        updated_at = _dt.datetime.fromisoformat(updated_at_str.replace("Z", "+00:00").split("+")[0])
                        age_seconds = (_dt.datetime.utcnow() - updated_at).total_seconds()
                        if age_seconds > _STALE_JOB_TIMEOUT_SECONDS:
                            job_is_stale = True
                            logger.warning(
                                "[STALE JOB] Job %s has been '%s' for %.0fs (>%ds) — treating as dead | file=%s",
                                existing_job['job_id'], existing_job['status'], age_seconds,
                                _STALE_JOB_TIMEOUT_SECONDS, safe_name,
                            )
                            # Mark the stale job as failed in the DB so it no longer blocks
                            try:
                                update_job(
                                    DB_PATH, existing_job['job_id'],
                                    status="failed",
                                    error_message="Job timed out (server restart or crash detected)."
                                )
                            except Exception as _upd_err:
                                logger.error("Failed to mark stale job as failed: %s", _upd_err)
                    except Exception:
                        pass
                if not job_is_stale:
                    is_job_active = True
        except Exception:
            pass

        if cnt == 0 and not is_job_active:
            # [STALE DOCUMENT DETECTED]
            stale_detected_msg = (
                f"\n[STALE DOCUMENT DETECTED]\n"
                f"document_id={existing_doc['id']}\n"
                f"file_name={safe_name}\n"
                f"reason=zero_chunks\n"
            )
            logger.info(stale_detected_msg)
            print(stale_detected_msg)

            sqlite_cleanup = "success"
            qdrant_cleanup = "success"

            try:
                # delete stale chunks (if any)
                from RAG.db import get_db_connection
                conn = get_db_connection(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks WHERE document_id = ?", (existing_doc["id"],))
                # delete stale job metadata
                cursor.execute("DELETE FROM ingestion_jobs WHERE file_hash = ?", (file_hash,))
                conn.commit()
                conn.close()

                # delete stale document record
                from RAG.db import delete_document
                delete_document(DB_PATH, existing_doc["id"])
            except Exception as exc:
                sqlite_cleanup = "failure"
                logger.error("Failed to clean up stale SQLite records: %s", exc)

            try:
                # delete stale vector records
                from RAG.vector_store import delete_vectors_by_doc
                delete_vectors_by_doc(VECTOR_DB_PATH, existing_doc["id"])
            except Exception as exc:
                qdrant_cleanup = "failure"
                logger.error("Failed to clean up stale Qdrant vector records: %s", exc)

            # [STALE CLEANUP]
            stale_cleanup_msg = (
                f"\n[STALE CLEANUP]\n"
                f"document_id={existing_doc['id']}\n"
                f"sqlite_cleanup={sqlite_cleanup}\n"
                f"qdrant_cleanup={qdrant_cleanup}\n"
            )
            logger.info(stale_cleanup_msg)
            print(stale_cleanup_msg)

            # [REINGEST ALLOWED]
            reingest_msg = (
                f"\n[REINGEST ALLOWED]\n"
                f"document_id_old={existing_doc['id']}\n"
                f"new_ingestion_started=true\n"
            )
            logger.info(reingest_msg)
            print(reingest_msg)

            existing_doc = None

    if existing_doc:
        logger.info(
            "[DUPLICATE] File already ingested | file=%s | doc_id=%d",
            safe_name, existing_doc["id"],
        )
        # [DUPLICATE CHECK] logging as requested
        try:
            from RAG.db import get_db_connection
            conn = get_db_connection(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (existing_doc["id"],))
            cnt = cursor.fetchone()[0]
            conn.close()
            dup_msg = (
                f"\n[DUPLICATE CHECK]\n"
                f"document_id={existing_doc['id']}\n"
                f"filename={safe_name}\n"
                f"chunk_count={cnt}\n"
            )
            logger.info(dup_msg)
            print(dup_msg)
        except Exception as exc:
            logger.error("Failed to retrieve duplicate chunk count: %s", exc)

        # Update last_active_at timestamp for duplicate uploads
        try:
            update_document_active_timestamp(DB_PATH, existing_doc["id"])
        except Exception as exc:
            logger.error("Failed to update active document timestamp for duplicate: %s", exc)

        # Return a stable synthetic job_id derived from the hash so callers get consistent IDs
        synthetic_job_id = f"dup-{file_hash[:16]}"
        return UploadAcceptedResponse(
            status="already_ingested",
            job_id=synthetic_job_id,
            file_name=safe_name,
            file_type=ext,
            message=(
                f"Document already exists in the knowledge base "
                f"(document_id={existing_doc['id']}). "
                f"You can query it immediately."
            ),
            document_id=existing_doc["id"],
        )

    # --- 6. Check for an in-progress / recent job for this hash ---
    logger.info("UPLOAD file validation completed | duplicate check starting")
    try:
        existing_job = get_job_by_hash(DB_PATH, file_hash)
    except Exception:
        existing_job = None

    if existing_job and existing_job["status"] in ("pending", "running"):
        logger.info(
            "[DUPLICATE JOB] Ingestion already in progress | job_id=%s | file=%s",
            existing_job["job_id"], safe_name,
        )
        return UploadAcceptedResponse(
            status="accepted",
            job_id=existing_job["job_id"],
            file_name=safe_name,
            file_type=ext,
            message="Ingestion already in progress for this file. Poll GET /jobs/{job_id} for status.",
        )

    # --- 7. Create job record & start background thread ---
    job_id = str(uuid.uuid4())
    create_job(DB_PATH, job_id, safe_name, file_hash)

    # Retrieve and propagate the request identifier to the background worker thread
    request_id = request_id_var.get() or str(uuid.uuid4())

    t = threading.Thread(
        target=_run_ingestion_job,
        args=(job_id, file_bytes, safe_name, ext, file_hash, request_id),
        daemon=True,
        name=f"ingest-{job_id[:8]}",
    )
    t.start()

    logger.info(
        "[UPLOAD ACCEPTED] job_id=%s | file=%s | background thread started",
        job_id, safe_name,
    )

    return UploadAcceptedResponse(
        status="accepted",
        job_id=job_id,
        file_name=safe_name,
        file_type=ext,
        message=(
            f"File accepted and ingestion started in the background. "
            f"Poll GET /jobs/{job_id} to check progress."
        ),
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  — Job Status Polling
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Ingestion Job Status",
    description=(
        "Poll this endpoint after `POST /upload` to check ingestion progress.\n\n"
        "**Status values:**\n"
        "- `pending` — queued, not yet started\n"
        "- `running` — extraction pipeline is active\n"
        "- `success` — fully ingested, ready to query\n"
        "- `failed` — pipeline error (see `error_message`)\n"
        "- `already_ingested` — file was already in the knowledge base\n\n"
        "Once status is `success`, use `POST /query` with the returned `document_id`."
    ),
    tags=["Upload"],
)
async def get_job_status(job_id: str):
    """Returns the current status of an ingestion job."""
    # Handle synthetic IDs for already-ingested duplicates
    if job_id.startswith("dup-"):
        hash_prefix = job_id[4:]
        doc_id = None
        doc_name = "unknown"
        
        try:
            from RAG.db import get_db_connection
            conn = get_db_connection(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename FROM documents WHERE file_hash LIKE ? LIMIT 1", (f"{hash_prefix}%",))
            row = cursor.fetchone()
            if row:
                doc_id = row["id"]
                doc_name = row["filename"]
            conn.close()
        except Exception as exc:
            logger.error("Failed to lookup duplicate document metadata for %s: %s", job_id, exc)

        return JobStatusResponse(
            job_id=job_id,
            status="already_ingested",
            file_name=doc_name,
            document_id=doc_id,
            total_chunks=0,
        )

    try:
        job = get_job(DB_PATH, job_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {exc}",
        )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        file_name=job["file_name"],
        document_id=job.get("document_id"),
        pages_processed=job.get("pages_processed", 0),
        text_chunks=job.get("text_chunks", 0),
        table_chunks=job.get("table_chunks", 0),
        image_chunks=job.get("image_chunks", 0),
        total_chunks=job.get("total_chunks", 0),
        error_message=job.get("error_message"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )
