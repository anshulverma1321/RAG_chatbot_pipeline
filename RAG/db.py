import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from RAG.exceptions import DatabaseError

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key enforcement (disabled by default in Python's sqlite3 module)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_db_connection", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e

def init_db(db_path: str):
    """Initializes the SQLite schema for tracking documents and chunks."""
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            total_pages INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create chunks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id INTEGER,
            page_number INTEGER,
            chunk_type TEXT CHECK(chunk_type IN ('text', 'table', 'image')),
            content TEXT NOT NULL,
            sibling_order INTEGER,
            asset_type TEXT,
            classification_type TEXT,
            extractor_used TEXT,
            document_name TEXT,
            metadata TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        """)

        # Check if existing columns are missing (schema migration)
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [row[1] for row in cursor.fetchall()]
        if columns:
            new_cols = {
                "asset_type": "TEXT",
                "classification_type": "TEXT",
                "extractor_used": "TEXT",
                "document_name": "TEXT",
                "metadata": "TEXT"
            }
            for col_name, col_type in new_cols.items():
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")

        # Check if last_active_at exists in documents (schema migration)
        cursor.execute("PRAGMA table_info(documents)")
        doc_columns = [row[1] for row in cursor.fetchall()]
        if doc_columns:
            if "last_active_at" not in doc_columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN last_active_at TIMESTAMP")
        
        # Create ingestion_jobs table for async background processing
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            document_id INTEGER,
            pages_processed INTEGER DEFAULT 0,
            text_chunks INTEGER DEFAULT 0,
            table_chunks INTEGER DEFAULT 0,
            image_chunks INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create indexes for fast retrieval
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (document_id, page_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_hash ON ingestion_jobs (file_hash)")

        conn.commit()
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "init_db", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        if conn:
            conn.close()

def add_document(db_path: str, filename: str, file_hash: str, total_pages: int) -> int:
    """Adds a document metadata record and returns its ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO documents (filename, file_hash, total_pages) VALUES (?, ?, ?)",
            (filename, file_hash, total_pages)
        )
        conn.commit()
        doc_id = cursor.lastrowid
        return doc_id
    except sqlite3.IntegrityError:
        # Document already exists, retrieve its ID
        try:
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row:
                return row['id']
            raise DatabaseError("Metadata database operation failed.")
        except sqlite3.Error as e:
            from RAG.logger import log_error
            log_error("RAG.db", "add_document_retrieve_id", type(e).__name__, str(e))
            raise DatabaseError("Metadata database operation failed.") from e
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "add_document", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def add_chunks(db_path: str, chunks: List[Dict[str, Any]]):
    """Bulk inserts chunks into the chunks table."""
    import json
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        sanitized_chunks = []
        for chunk in chunks:
            # Handle metadata dict to JSON string conversion
            meta_val = chunk.get("metadata")
            if isinstance(meta_val, dict):
                meta_val = json.dumps(meta_val)
            
            sanitized = {
                "id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "page_number": chunk.get("page_number"),
                "chunk_type": chunk.get("chunk_type") or chunk.get("asset_type"),
                "content": chunk.get("content"),
                "sibling_order": chunk.get("sibling_order"),
                "asset_type": chunk.get("asset_type") or chunk.get("chunk_type"),
                "classification_type": chunk.get("classification_type"),
                "extractor_used": chunk.get("extractor_used"),
                "document_name": chunk.get("document_name"),
                "metadata": meta_val
            }
            sanitized_chunks.append(sanitized)

        cursor.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id, document_id, page_number, chunk_type, content, sibling_order,
                asset_type, classification_type, extractor_used, document_name, metadata
            )
            VALUES (
                :id, :document_id, :page_number, :chunk_type, :content, :sibling_order,
                :asset_type, :classification_type, :extractor_used, :document_name, :metadata
            )
            """,
            sanitized_chunks
        )
        conn.commit()
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "add_chunks", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def get_document_by_hash(db_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieves document record by file hash."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_document_by_hash", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def get_chunk(db_path: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a chunk by its ID, resolving document_name from the documents table if needed."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT c.*, d.filename AS doc_filename 
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
        """, (chunk_id,))
        row = cursor.fetchone()
        if row:
            res = dict(row)
            if not res.get("document_name"):
                res["document_name"] = res.get("doc_filename") or "Unknown"
            return res
        return None
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_chunk", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def get_sibling_chunks(db_path: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
    """Retrieves all chunks belonging to a specific page of a document, ordered chronologically."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? AND page_number = ? ORDER BY sibling_order",
            (document_id, page_number)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_sibling_chunks", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def list_documents(db_path: str) -> List[Dict[str, Any]]:
    """Lists all uploaded documents."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "list_documents", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def delete_document(db_path: str, document_id: int):
    """Deletes a document and cascade deletes its chunks."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "delete_document", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()

def update_document_active_timestamp(db_path: str, document_id: int):
    """Updates the last_active_at timestamp for a document to make it the active document."""
    import datetime
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        # Get timestamp before
        cursor.execute("SELECT last_active_at FROM documents WHERE id = ?", (document_id,))
        row_before = cursor.fetchone()
        t_before = row_before[0] if row_before else "None"

        now_str = datetime.datetime.utcnow().isoformat()
        cursor.execute(
            "UPDATE documents SET last_active_at = ? WHERE id = ?",
            (now_str, document_id)
        )
        rows_updated = cursor.rowcount
        conn.commit()
        commit_success = True

        # Get timestamp after
        cursor.execute("SELECT last_active_at FROM documents WHERE id = ?", (document_id,))
        row_after = cursor.fetchone()
        t_after = row_after[0] if row_after else "None"

        # Log ACTIVE UPDATE block as requested
        import logging
        db_logger = logging.getLogger("RAG.db")
        msg = (
            f"\n[ACTIVE UPDATE]\n"
            f"document_id={document_id}\n"
            f"timestamp_before={t_before}\n"
            f"timestamp_after={t_after}\n"
            f"rows_updated={rows_updated}\n"
            f"commit_success={commit_success}\n"
        )
        db_logger.info(msg)
        print(msg)
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "update_document_active_timestamp", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ingestion Jobs — async job tracking
# ---------------------------------------------------------------------------

def create_job(db_path: str, job_id: str, file_name: str, file_hash: str) -> None:
    """Creates a new ingestion job record with status='pending'."""
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO ingestion_jobs (job_id, file_name, file_hash, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (job_id, file_name, file_hash),
        )
        conn.commit()
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "create_job", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()


def update_job(
    db_path: str,
    job_id: str,
    status: str,
    document_id: Optional[int] = None,
    pages_processed: int = 0,
    text_chunks: int = 0,
    table_chunks: int = 0,
    image_chunks: int = 0,
    total_chunks: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Updates an ingestion job's status and result fields."""
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE ingestion_jobs SET
                status = ?,
                document_id = ?,
                pages_processed = ?,
                text_chunks = ?,
                table_chunks = ?,
                image_chunks = ?,
                total_chunks = ?,
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (
                status,
                document_id,
                pages_processed,
                text_chunks,
                table_chunks,
                image_chunks,
                total_chunks,
                error_message,
                job_id,
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "update_job", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()


def get_job(db_path: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an ingestion job by job_id."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_job", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()


def get_job_by_hash(db_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Returns the most recent job for a given file hash (any status)."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE file_hash = ? ORDER BY created_at DESC LIMIT 1",
            (file_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        from RAG.logger import log_error
        log_error("RAG.db", "get_job_by_hash", type(e).__name__, str(e))
        raise DatabaseError("Metadata database operation failed.") from e
    finally:
        conn.close()
