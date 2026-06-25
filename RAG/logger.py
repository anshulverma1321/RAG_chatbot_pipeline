import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import traceback
import json
import contextvars
from typing import Optional, Dict, Any, List

# Context variables for request tracing and observability metadata
request_id_var = contextvars.ContextVar("request_id", default=None)
document_id_var = contextvars.ContextVar("document_id", default=None)
file_name_var = contextvars.ContextVar("file_name", default=None)
execution_stage_var = contextvars.ContextVar("execution_stage", default=None)
timings_var = contextvars.ContextVar("timings", default=None)

# Timings helpers for ingestion pipeline
def init_performance_timings():
    """Initializes performance timing dictionary in the current context."""
    timings_var.set({
        "routing_time": 0.0,
        "classification_time": 0.0,
        "extraction_time": 0.0,
        "normalization_time": 0.0,
        "ingestion_time": 0.0
    })

def record_performance_timing(stage: str, duration: float):
    """Accumulates duration for a specific execution stage."""
    t = timings_var.get()
    if t is not None:
        t[stage] = t.get(stage, 0.0) + duration

def get_performance_timings() -> Optional[Dict[str, float]]:
    """Retrieves accumulated performance timings."""
    return timings_var.get()

# Resolve base project directory and ensure logs directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Custom formatter to output JSON structured logs."""
    def format(self, record):
        doc_id = getattr(record, "document_id", None) or document_id_var.get()
        f_name = getattr(record, "file_name", None) or file_name_var.get()
        stage = getattr(record, "stage", None) or execution_stage_var.get() or ""
        req_id = getattr(record, "request_id", None) or request_id_var.get() or ""

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "request_id": req_id,
            "document_id": doc_id,
            "file_name": f_name,
            "stage": stage,
            "message": record.getMessage()
        }
        
        # Include custom metrics if available
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
            
        if record.exc_info:
            log_data["traceback"] = "".join(traceback.format_exception(*record.exc_info))
            
        return json.dumps(log_data)

def _setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Creates a configured logger with RotatingFileHandler and JSONFormatter."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Reset existing handlers to prevent duplicates
    logger.handlers = []
    
    # 10 MB rotation, keep 5 backups
    handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, log_file), 
        maxBytes=10*1024*1024, 
        backupCount=5, 
        encoding='utf-8'
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Also log any errors (ERROR/CRITICAL) to errors.log automatically
    if name != "errors":
        error_handler = RotatingFileHandler(
            os.path.join(LOGS_DIR, "errors.log"), 
            maxBytes=10*1024*1024, 
            backupCount=5, 
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)
        
    return logger

# Initialize specific loggers
api_logger = _setup_logger("api", "api.log")
upload_logger = _setup_logger("upload", "upload.log")
ingestion_logger = _setup_logger("ingestion", "ingestion.log")
retrieval_logger = _setup_logger("retrieval", "retrieval.log")
llm_logger = _setup_logger("llm", "llm.log")
errors_logger = _setup_logger("errors", "errors.log", logging.ERROR)
performance_logger = _setup_logger("performance", "performance.log")

# Legacy compatible aliases
app_logger = api_logger
chat_logger = retrieval_logger

def is_debug_mode() -> bool:
    """Checks if DEBUG_MODE is enabled in environment settings."""
    return os.environ.get("DEBUG_MODE", "false").lower() == "true"

def log_ingestion_start(file_name: str):
    """Logs the start of the ingestion process."""
    file_name_var.set(file_name)
    execution_stage_var.set("ingestion")
    msg = f"Ingestion started for file: {file_name}"
    ingestion_logger.info(msg)
    upload_logger.info(msg)

def log_ingestion_success(file_name: str, pages: int, chunks_created: int, images_found: int, tables_found: int):
    """Logs successful ingestion with metadata counters."""
    file_name_var.set(file_name)
    execution_stage_var.set("ingestion")
    msg = (
        f"Ingestion successful | file={file_name} | pages={pages} | "
        f"chunks={chunks_created} | images={images_found} | tables={tables_found}"
    )
    ingestion_logger.info(msg)
    upload_logger.info(msg)

def log_query(query_text: str):
    """Logs the user query received."""
    execution_stage_var.set("query")
    retrieval_logger.info(f"Query received: {query_text}")

def log_retrieval(query_text: str, results: list):
    """Logs retrieved chunks and similarity scores."""
    execution_stage_var.set("retrieval")
    summary = []
    for r in results:
        payload = r.get('payload', {})
        filename = payload.get('filename', f"Doc-{payload.get('document_id')}")
        page_num = payload.get('page_number', 'unknown')
        score = r.get('score', 0.0)
        summary.append(f"{filename} Page {page_num} (Score: {score:.2f})")
    
    retrieval_logger.info(f"Retrieval complete | matches: {', '.join(summary)}")

def log_context(files_used: list, pages: list, total_context_length: int):
    """Logs details of context assembled for LLM generation."""
    execution_stage_var.set("context_assembly")
    files_str = ", ".join(sorted(list(set(files_used))))
    pages_str = ", ".join(map(str, sorted(list(set(pages)))))
    msg = f"Context assembled | files=[{files_str}] | pages=[{pages_str}] | length={total_context_length}"
    retrieval_logger.info(msg)
    llm_logger.info(msg)

def log_response(response_length: int, generation_time: float):
    """Logs response metadata from the LLM."""
    execution_stage_var.set("llm_generation")
    llm_logger.info(f"LLM Response generated | length={response_length} | generation_time={generation_time:.2f}s")

def log_whisper(audio_duration: float, recognized_text: str, status: str):
    """Logs speech transcription metrics."""
    execution_stage_var.set("speech_to_text")
    api_logger.info(f"Whisper transcription | duration={audio_duration:.2f}s | status={status} | text_len={len(recognized_text)}")

def log_piper(text_length: int, voice: str, audio_generated: str):
    """Logs Piper speech synthesis metrics."""
    execution_stage_var.set("text_to_speech")
    api_logger.info(f"Piper synthesis | text_len={text_length} | voice={voice} | file={audio_generated}")

def log_error(module: str, function: str, error_type: str, error_message: str):
    """Logs structured error information."""
    msg = f"Error in {module}.{function} | type={error_type} | message={error_message}"
    errors_logger.error(msg)

def log_performance(metrics: dict):
    """Logs performance timing metrics."""
    performance_logger.info("Performance Timings", extra={"metrics": metrics})

def log_system_check(db_path: str, vector_db_path: str):
    """Performs and logs startup diagnostics check for components."""
    py_ver = sys.version.split()[0]
    gemini_status = "Connected" if os.environ.get("GEMINI_API_KEY") else "Disconnected (Missing Key)"
    
    api_logger.info(f"System Check | Python={py_ver} | Gemini={gemini_status} | DB={db_path} | VectorDB={vector_db_path}")
    print(f"System Check complete. Python: {py_ver}, Gemini: {gemini_status}")

def configure_logger_routing():
    """Maps module-level loggers to the specific rotating JSON file handlers."""
    def set_logger_handlers(logger_name: str, source_logger_name: str):
        target = logging.getLogger(logger_name)
        source = logging.getLogger(source_logger_name)
        target.handlers = list(source.handlers)
        target.setLevel(source.level)
        target.propagate = False

    # Route routes and content router to upload logs
    set_logger_handlers("RAG.routes.upload", "upload")
    set_logger_handlers("RAG.routes.process", "upload")
    set_logger_handlers("RAG.content_router", "upload")
    
    # Route main pipelines to ingestion logs
    set_logger_handlers("RAG.ingestion", "ingestion")
    set_logger_handlers("RAG.document_orchestrator", "ingestion")
    
    # Route service extractors to ingestion logs
    set_logger_handlers("RAG.services.chart_knowledge_extractor", "ingestion")
    set_logger_handlers("RAG.services.diagram_knowledge_extractor", "ingestion")
    set_logger_handlers("RAG.services.image_classifier", "ingestion")
    set_logger_handlers("RAG.services.image_intelligence", "ingestion")
    set_logger_handlers("RAG.services.table_classifier", "ingestion")
    set_logger_handlers("RAG.services.table_intelligence", "ingestion")
    set_logger_handlers("RAG.services.text_knowledge_extractor", "ingestion")
    
    # Route query engine to retrieval logs
    set_logger_handlers("RAG.query_engine", "retrieval")
    
    # Route app and uvicorn loggers to api logs
    set_logger_handlers("RAG.app", "api")
    set_logger_handlers("uvicorn", "api")
    set_logger_handlers("uvicorn.access", "api")
    set_logger_handlers("uvicorn.error", "api")

# Execute logger routing configuration automatically on module import
configure_logger_routing()
