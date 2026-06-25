import os
import logging
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from RAG.db import init_db
from RAG.vector_store import init_vector_store
from RAG.ingestion import process_pdf
from RAG.query_engine import execute_rag_query
from RAG.routes.validation import router as validation_router
from RAG.routes.upload import router as upload_router
from RAG.routes.process import router as process_router

# Compute database and vector store paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rag_tool.db")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "qdrant")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Initialize Directories and Databases
os.makedirs(TEMP_DIR, exist_ok=True)
init_db(DB_PATH)
init_vector_store(VECTOR_DB_PATH)

# ---------------------------------------------------------------------------
# Startup Cleanup — runs once on every server start
# ---------------------------------------------------------------------------

def _startup_cleanup():
    """
    Performs three critical cleanup tasks on every server restart:

    1. TEMP FILE PURGE: Deletes all leftover tmp*.pdf files from the previous
       server session.  These accumulate because background ingestion threads
       die on SIGKILL/restart before their `finally: os.remove(temp_path)` runs.

    2. STALE JOB RECOVERY: Marks every job still in 'running' or 'pending'
       state as 'failed'.  The background threads that owned those jobs no
       longer exist after a restart, so the jobs will never complete on their
       own.  Marking them 'failed' allows the stale-document cleanup logic in
       POST /upload to detect and re-ingest their documents automatically.

    3. STALE DOCUMENT CLEANUP: Deletes document records that have 0 chunks and
       no active job.  This ensures re-uploading the same file always triggers
       a fresh full ingestion instead of returning 'already_ingested'.
    """
    import sqlite3 as _sqlite3
    import glob as _glob

    # 1. Purge orphaned temp files
    purged = 0
    try:
        pattern = os.path.join(TEMP_DIR, "tmp*")
        for f in _glob.glob(pattern):
            try:
                os.remove(f)
                purged += 1
            except Exception:
                pass
        if purged:
            logger.info("[STARTUP] Purged %d orphaned temp file(s) from %s", purged, TEMP_DIR)
    except Exception as e:
        logger.warning("[STARTUP] Temp-file purge failed: %s", e)

    # 2. Mark stuck running/pending jobs as failed
    marked_failed = 0
    try:
        conn = _sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed',
                error_message = 'Server restart detected — background thread terminated.',
                updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('running', 'pending')
            """
        )
        marked_failed = cursor.rowcount
        conn.commit()
        conn.close()
        if marked_failed:
            logger.info(
                "[STARTUP] Marked %d stuck job(s) as failed (server restart recovery)",
                marked_failed,
            )
    except Exception as e:
        logger.warning("[STARTUP] Stale-job recovery failed: %s", e)

    # 3. Delete stale document records with 0 chunks whose job is failed/missing.
    #    These are documents whose ingestion was interrupted mid-way (e.g. server
    #    restart).  Without this cleanup, re-uploading the same file returns
    #    "already_ingested" and the upload.py stale-check only fires when the
    #    user explicitly re-uploads.  By cleaning up here we guarantee that the
    #    next upload always triggers a fresh full ingestion automatically.
    deleted_stale = 0
    try:
        conn = _sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Find documents with zero chunks that have no active (pending/running) job
        cursor.execute(
            """
            SELECT d.id, d.filename
            FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
              AND NOT EXISTS (
                    SELECT 1 FROM ingestion_jobs j
                    WHERE j.file_hash = d.file_hash
                      AND j.status IN ('pending', 'running')
              )
            """
        )
        stale_docs = cursor.fetchall()
        for doc_id, filename in stale_docs:
            cursor.execute(
                "DELETE FROM ingestion_jobs WHERE file_hash = (SELECT file_hash FROM documents WHERE id=?)",
                (doc_id,)
            )
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted_stale += 1
            logger.info(
                "[STARTUP] Deleted stale 0-chunk document | doc_id=%d | file=%s",
                doc_id, filename,
            )
        conn.commit()
        conn.close()
        if deleted_stale:
            logger.info(
                "[STARTUP] Cleaned up %d stale document record(s) with zero chunks",
                deleted_stale,
            )
    except Exception as e:
        logger.warning("[STARTUP] Stale document cleanup failed: %s", e)


_startup_cleanup()

app = FastAPI(
    title="Multimodal Document Intelligence RAG API",
    description=(
        "**Production-ready Multimodal RAG Platform**\n\n"
        "Upload any supported document (PDF, PNG, JPG, JPEG, CSV, XLSX, XLS) via `POST /upload` "
        "and query your knowledge base via `POST /query`.\n\n"
        "The system automatically routes, processes, extracts multimodal knowledge, "
        "generates embeddings, and stores everything into SQLite + Qdrant.\n\n"
        "**Public Endpoints:** `POST /upload` · `POST /query`"
    ),
    version="8.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins so the Swagger UI, frontend apps, and local tools can
# reach the API without "Failed to fetch" / CORS errors.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # all origins (tighten in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import uuid
import time
from fastapi import Request
from RAG.logger import request_id_var, execution_stage_var, api_logger, errors_logger

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Generate unique request tracing identifier
    request_id = str(uuid.uuid4())
    req_token = request_id_var.set(request_id)
    stage_token = execution_stage_var.set("api_request")
    
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Access log formatting as per guidelines:
        # HTTP method, route, status code, execution time, client IP
        msg = f"{request.method} {request.url.path}\n{response.status_code}\n{process_time:.2f} sec\nClient IP: {client_ip}"
        api_logger.info(msg)
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        msg = f"{request.method} {request.url.path}\n500\n{process_time:.2f} sec\nClient IP: {client_ip}"
        api_logger.error(msg)
        # Preserve full traceback
        errors_logger.exception(f"Unhandled exception during API request to {request.url.path}: {exc}")
        raise
    finally:
        request_id_var.reset(req_token)
        execution_stage_var.reset(stage_token)


# Register validation router
app.include_router(validation_router, prefix="/validation", tags=["Validation APIs"])

# Register universal upload router (Phase 2 - routing only)
app.include_router(upload_router, tags=["Upload"])

# Register universal process router (Phase 3 - routing & processing)
app.include_router(process_router, tags=["Process"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="The search query (1-4000 chars).")
    document_ids: Optional[List[int]] = None

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Multimodal Document Intelligence RAG Tool",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "gemini_api_key_configured": api_key_set
    }

@app.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,  # Phase 8: Internal legacy endpoint — hidden from Swagger docs
    summary="Ingest PDF Document",
    description="Uploads a PDF file and processes it page-by-page for text, tables, and images. Inserts extracted chunks into SQLite and Qdrant vector store.",
    tags=["Ingestion"],
)
async def ingest_pdf(file: UploadFile = File(...)):
    """Full PDF ingestion: extracts text, tables, images → SQLite + Qdrant."""
    logger.info("[INGEST START] POST /ingest received | filename=%s", file.filename)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )
        
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured on the server."
        )
        
    # Sanitize filename to prevent path traversal attacks
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename."
        )
    temp_file_path = os.path.join(TEMP_DIR, safe_name)
    try:
        # Save file to temp path
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Reject files larger than 50 MB
        MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
        if os.path.getsize(temp_file_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 50 MB upload limit."
            )
            
        # Process document
        doc_id, msg = process_pdf(temp_file_path, DB_PATH, VECTOR_DB_PATH)
        return {
            "document_id": doc_id,
            "message": msg
        }
    except Exception as e:
        from RAG.exceptions import DuplicateDocumentError
        if isinstance(e, DuplicateDocumentError):
            # Return HTTP 200 with already_ingested status (not a hard error)
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post(
    "/query",
    summary="Query Knowledge Base",
    description=(
        "Executes a grounded semantic search query against all ingested documents.\n\n"
        "Optionally filter by `document_ids` to restrict the search to specific documents.\n\n"
        "**Workflow:** Upload Once → Ingest Once → Query Unlimited Times"
    ),
    tags=["Query"],
)
async def query_rag(request: QueryRequest):
    """Executes a grounded query against the processed documents."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured on the server."
        )
        
    try:
        answer = execute_rag_query(
            query=request.query,
            db_path=DB_PATH,
            vector_db_path=VECTOR_DB_PATH,
            document_ids=request.document_ids,
            top_k=5
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge base: {str(e)}"
        )
