import os
import time
import logging
import tempfile
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from RAG.content_router import route_content, SUPPORTED_EXTENSIONS, RoutingMetadata
from RAG.routes.upload import _build_processing_strategy, ClassificationDetail, RoutingResult

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Table extractors — Gemini-only, no PaddleOCR dependency. Safe to import at module level.
from RAG.services.table_intelligence import (
    extract_simple_table,
    extract_comparison_table,
    extract_financial_table,
    extract_statistical_table,
    extract_timeseries_table,
    extract_unknown_table,
)

# Image extractors and orchestrator — imported at module level so unit-test mock patches
# targeting RAG.routes.process.<name> work correctly. PaddleOCR is still lazy-loaded
# inside the individual extractor functions themselves (image_intelligence.py etc.),
# so server startup overhead is unchanged.
from RAG.services.text_knowledge_extractor import extract_text_knowledge
from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
from RAG.ingestion import describe_image_with_gemini
from RAG.document_orchestrator import orchestrate_pdf
from RAG.routes.validation import (
    run_visual_understanding_logic_on_bytes,
    parse_table_file_to_markdown,
)



class ProcessTiming(BaseModel):
    """Execution timing measurements in seconds."""
    total_seconds: float = Field(..., description="Total execution time in seconds.")
    routing_seconds: float = Field(..., description="Time spent in file routing and classification.")
    processing_seconds: float = Field(..., description="Time spent executing the extraction pipeline.")


class ProcessResponse(BaseModel):
    """Top-level response envelope for POST /process."""
    routing_metadata: RoutingResult = Field(..., description="Standardized routing decisions.")
    selected_extractor: str = Field(..., description="The name of the pipeline extractor function that was executed.")
    extracted_knowledge: Dict[str, Any] = Field(..., description="The structured knowledge payload returned by the extractor.")
    rich_text_representation: str = Field(..., description="Markdown-formatted rich text representation of the extracted knowledge.")
    execution_timing: ProcessTiming = Field(..., description="Execution duration metrics.")


@router.post(
    "/process",
    response_model=Union[ProcessResponse, Dict[str, str]],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,  # Phase 8: Internal layer only — hidden from Swagger docs
    summary="Universal File Routing & Processing",
    description=(
        "**Phase 3 — Universal Processing API**\n\n"
        "Accepts an uploaded file, routes it dynamically using the content router, "
        "and automatically triggers the matching specialized extraction pipeline.\n\n"
        "**No ingestion or DB updates are performed.**\n\n"
        "**Image routing:**\n"
        "- `TEXT_IMAGE` → `extract_text_knowledge()`\n"
        "- `CHART` → `extract_chart_knowledge()`\n"
        "- `DIAGRAM` → `extract_diagram_knowledge()`\n"
        "- `MIXED` → `run_visual_understanding_logic()`\n"
        "- `NATURAL_IMAGE` → `describe_image_with_gemini()`\n\n"
        "**Table routing:**\n"
        "- `TABLE_SIMPLE` → `extract_simple_table()`\n"
        "- `TABLE_COMPARISON` → `extract_comparison_table()`\n"
        "- `TABLE_FINANCIAL` → `extract_financial_table()`\n"
        "- `TABLE_STATISTICAL` → `extract_statistical_table()`\n"
        "- `TABLE_TIMESERIES` → `extract_timeseries_table()`\n\n"
        "**PDF files:**\n"
        "- Returns `{\"status\": \"pdf_processing_not_implemented\"}` immediately."
    ),
    tags=["Process"],
)
async def universal_process(file: UploadFile = File(..., description="The file to route and process.")):
    """
    Receives an uploaded file, runs routing metadata checks, routes to the correct
    intelligence pipeline extractor, and returns extraction results.
    """
    t_start = time.perf_counter()
    logger.info("[PROCESS START] POST /process received | filename=%s", file.filename)

    # --- 1. Basic Filename and Extension Validation ---
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

    ext = os.path.splitext(safe_name)[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS)).upper()}."
            ),
        )

    # --- 2. Read File Bytes & Validate size/emptiness ---
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file %s: %s", safe_name, exc)
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

    # --- 3. Routing & Classification Phase ---
    t_route_start = time.perf_counter()
    try:
        routing_meta: RoutingMetadata = route_content(
            file_name=safe_name,
            file_bytes=file_bytes,
            mime_type=file.content_type or None,
        )
    except Exception as exc:
        logger.exception("content_router.route_content failed for %s: %s", safe_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failed: {exc}",
        )
    t_route_end = time.perf_counter()
    routing_seconds = round(t_route_end - t_route_start, 4)

    # --- 4. PDF Orchestration ---
    if routing_meta.primary_category == "document" or routing_meta.file_extension == "pdf":
        t_proc_start = time.perf_counter()
        
        # Save file to temp path
        suffix = f".{routing_meta.file_extension}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            chunks = orchestrate_pdf(temp_path)
        except Exception as exc:
            logger.exception("orchestrate_pdf failed for %s: %s", safe_name, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF orchestration failed: {exc}",
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        t_proc_end = time.perf_counter()
        processing_seconds = round(t_proc_end - t_proc_start, 4)
        total_seconds = round(time.perf_counter() - t_start, 4)

        # Formulate timing
        timing = ProcessTiming(
            total_seconds=total_seconds,
            routing_seconds=routing_seconds,
            processing_seconds=processing_seconds,
        )

        # Formulate rich text representation (concatenated Markdown)
        rich_text_representation = "\n\n".join([c.rich_text_representation for c in chunks])

        # Formulate routing result
        processing_strategy = _build_processing_strategy(
            primary_category=routing_meta.primary_category,
            classification=routing_meta.classification,
        )

        classification_detail: Optional[ClassificationDetail] = None
        if routing_meta.classification:
            classification_detail = ClassificationDetail(
                type=routing_meta.classification.get("type", ""),
                confidence=routing_meta.classification.get("confidence", 0.0),
                reason=routing_meta.classification.get("reason", ""),
            )

        routing_result = RoutingResult(
            file_name=routing_meta.file_name,
            file_extension=routing_meta.file_extension,
            mime_type=routing_meta.mime_type,
            primary_category=routing_meta.primary_category,
            is_supported=routing_meta.is_supported,
            suggested_route=routing_meta.suggested_route,
            suggested_extractor=routing_meta.suggested_extractor,
            processing_strategy=processing_strategy,
            classification=classification_detail,
        )

        return ProcessResponse(
            routing_metadata=routing_result,
            selected_extractor="document_orchestrator",
            extracted_knowledge={"chunks": [c.model_dump() for c in chunks]},
            rich_text_representation=rich_text_representation,
            execution_timing=timing,
        )

    # --- 5. Extraction & Execution Pipeline Phase ---
    t_proc_start = time.perf_counter()
    selected_extractor = ""
    extracted_knowledge = {}
    rich_text_representation = ""

    try:
        if routing_meta.primary_category == "image":
            # Extract classifier result type
            vlm_type = ""
            if routing_meta.classification:
                vlm_type = routing_meta.classification.get("type", "").upper()

            logger.info(
                "[IMAGE DETECTED] POST /process routing image | vlm_type=%s | size=%d bytes",
                vlm_type, len(file_bytes),
            )

            if vlm_type == "TEXT_IMAGE":
                selected_extractor = "extract_text_knowledge"
                res = extract_text_knowledge(file_bytes, routing_meta.mime_type)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif vlm_type == "CHART":
                selected_extractor = "extract_chart_knowledge"
                res = extract_chart_knowledge(file_bytes, routing_meta.mime_type)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif vlm_type == "DIAGRAM":
                selected_extractor = "extract_diagram_knowledge"
                res = extract_diagram_knowledge(file_bytes, routing_meta.mime_type)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif vlm_type == "MIXED":
                selected_extractor = "run_visual_understanding_logic"
                res = run_visual_understanding_logic_on_bytes(file_bytes, routing_meta.mime_type)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif vlm_type == "NATURAL_IMAGE":
                selected_extractor = "describe_image_with_gemini"
                desc = describe_image_with_gemini(file_bytes, routing_meta.mime_type)
                extracted_knowledge = {"summary": desc}
                rich_text_representation = f"# Image Description\n\n## Gemini Vision Summary:\n{desc}\n"
            else:
                # Fallback default image descriptor
                selected_extractor = "describe_image_with_gemini"
                desc = describe_image_with_gemini(file_bytes, routing_meta.mime_type)
                extracted_knowledge = {"summary": desc}
                rich_text_representation = f"# Image Description\n\n## Gemini Vision Summary:\n{desc}\n"

        elif routing_meta.primary_category == "spreadsheet":
            # Parse table data into Markdown representation
            suffix = f".{routing_meta.file_extension}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            try:
                table_markdown = parse_table_file_to_markdown(temp_path, safe_name)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

            # Extract classifier result type
            tbl_type = ""
            if routing_meta.classification:
                tbl_type = routing_meta.classification.get("type", "").upper()

            if tbl_type == "TABLE_SIMPLE":
                selected_extractor = "extract_simple_table"
                res = extract_simple_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif tbl_type == "TABLE_COMPARISON":
                selected_extractor = "extract_comparison_table"
                res = extract_comparison_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif tbl_type == "TABLE_FINANCIAL":
                selected_extractor = "extract_financial_table"
                res = extract_financial_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif tbl_type == "TABLE_STATISTICAL":
                selected_extractor = "extract_statistical_table"
                res = extract_statistical_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            elif tbl_type == "TABLE_TIMESERIES":
                selected_extractor = "extract_timeseries_table"
                res = extract_timeseries_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
            else:
                # Fallback table extractor
                selected_extractor = "extract_unknown_table"
                res = extract_unknown_table(table_markdown)
                rich_text_representation = res.pop("rich_text_representation", "")
                extracted_knowledge = res
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Routing category '{routing_meta.primary_category}' is unsupported for execution pipeline.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Extractor execution failed for %s: %s", safe_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        )

    t_proc_end = time.perf_counter()
    processing_seconds = round(t_proc_end - t_proc_start, 4)
    total_seconds = round(time.perf_counter() - t_start, 4)

    # --- 6. Formulate Standardized Response Metadata Envelope ---
    processing_strategy = _build_processing_strategy(
        primary_category=routing_meta.primary_category,
        classification=routing_meta.classification,
    )

    classification_detail: Optional[ClassificationDetail] = None
    if routing_meta.classification:
        classification_detail = ClassificationDetail(
            type=routing_meta.classification.get("type", ""),
            confidence=routing_meta.classification.get("confidence", 0.0),
            reason=routing_meta.classification.get("reason", ""),
        )

    routing_result = RoutingResult(
        file_name=routing_meta.file_name,
        file_extension=routing_meta.file_extension,
        mime_type=routing_meta.mime_type,
        primary_category=routing_meta.primary_category,
        is_supported=routing_meta.is_supported,
        suggested_route=routing_meta.suggested_route,
        suggested_extractor=routing_meta.suggested_extractor,
        processing_strategy=processing_strategy,
        classification=classification_detail,
    )

    timing = ProcessTiming(
        total_seconds=total_seconds,
        routing_seconds=routing_seconds,
        processing_seconds=processing_seconds,
    )

    return ProcessResponse(
        routing_metadata=routing_result,
        selected_extractor=selected_extractor,
        extracted_knowledge=extracted_knowledge,
        rich_text_representation=rich_text_representation,
        execution_timing=timing,
    )
