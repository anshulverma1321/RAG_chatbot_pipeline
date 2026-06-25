import os
import logging
import tempfile
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Supported file formats/extensions
SUPPORTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls"}

# Standard MIME type mappings for supported extensions
MIME_TYPE_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
}

# Primary content categories
CATEGORY_MAP = {
    "pdf": "document",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "csv": "spreadsheet",
    "xlsx": "spreadsheet",
    "xls": "spreadsheet",
}

# Pipeline names mapping
SUGGESTED_ROUTE_MAP = {
    "document": "pdf_pipeline",
    "image": "image_pipeline",
    "spreadsheet": "table_pipeline",
}

# Default extractor names (fallback / entrypoint function name)
DEFAULT_EXTRACTOR_MAP = {
    "pdf": "process_pdf",
    "png": "classify_image",
    "jpg": "classify_image",
    "jpeg": "classify_image",
    "csv": "classify_table",
    "xlsx": "classify_table",
    "xls": "classify_table",
}

# Image extractor mapping based on Gemini classification
IMAGE_EXTRACTOR_MAP = {
    "TEXT_IMAGE": "extract_text_knowledge",
    "CHART": "extract_chart_knowledge",
    "DIAGRAM": "extract_diagram_knowledge",
    "MIXED": "run_visual_understanding_logic",
    "NATURAL_IMAGE": "extract_natural_image_knowledge",
    "UNKNOWN": "describe_image_with_gemini",
}

# Table extractor mapping based on Gemini classification
TABLE_EXTRACTOR_MAP = {
    "TABLE_SIMPLE": "extract_simple_table",
    "TABLE_COMPARISON": "extract_comparison_table",
    "TABLE_FINANCIAL": "extract_financial_table",
    "TABLE_STATISTICAL": "extract_statistical_table",
    "TABLE_TIMESERIES": "extract_timeseries_table",
    "TABLE_UNKNOWN": "extract_unknown_table",
}

class RoutingMetadata(BaseModel):
    """Standardized metadata representation returned by the content router."""
    file_name: str = Field(..., description="Name of the file.")
    file_extension: str = Field(..., description="Normalized lowercase file extension without a leading dot.")
    mime_type: str = Field(..., description="MIME type of the file.")
    primary_category: str = Field(..., description="Primary content category: 'document', 'image', 'spreadsheet', or 'unknown'.")
    is_supported: bool = Field(..., description="Indicates if the format is supported.")
    suggested_route: str = Field(..., description="Recommended pipeline routing name.")
    suggested_extractor: str = Field(..., description="Name of the suggested extractor function.")
    classification: Optional[Dict[str, Any]] = Field(None, description="Detailed dynamic classification metadata, if content analysis was performed.")

def route_content(
    file_path: Optional[str] = None,
    file_name: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None
) -> RoutingMetadata:
    """
    Analyzes input file information and returns standardized routing metadata.
    
    If file content (file_bytes or an existing file_path) is provided, the function
    performs dynamic classification (VLM for images, LLM for tables) to return
    granular extraction routing decisions. Otherwise, it falls back to static extension-based routing.

    Args:
        file_path: Optional path to the file on disk.
        file_name: Optional name of the file (including extension).
        file_bytes: Optional raw bytes of the file.
        mime_type: Optional MIME type of the file.

    Returns:
        RoutingMetadata containing the standardized routing choices.
        
    Raises:
        ValueError: If none of file_path, file_name, or mime_type are provided.
    """
    if not file_path and not file_name and not mime_type:
        raise ValueError("At least one identifying parameter (file_path, file_name, or mime_type) must be provided.")

    # 1. Resolve file extension
    ext = ""
    if file_name:
        ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    elif file_path:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

    # Reverse lookup from MIME type if extension is still unresolved
    if not ext and mime_type:
        mime_lower = mime_type.lower()
        for k, v in MIME_TYPE_MAP.items():
            if v == mime_lower:
                ext = k
                break

    # 2. Resolve MIME type
    if not mime_type and ext:
        mime_type = MIME_TYPE_MAP.get(ext, "application/octet-stream")
    elif not mime_type:
        mime_type = "application/octet-stream"

    # 3. Resolve file name
    if not file_name:
        if file_path:
            file_name = os.path.basename(file_path)
        else:
            file_name = f"unknown_file.{ext}" if ext else "unknown_file"

    # 4. Check support and base fields
    is_supported = ext in SUPPORTED_EXTENSIONS
    classification = None

    if is_supported:
        primary_category = CATEGORY_MAP[ext]
        suggested_route = SUGGESTED_ROUTE_MAP[primary_category]
        suggested_extractor = DEFAULT_EXTRACTOR_MAP[ext]
    else:
        primary_category = "unknown"
        suggested_route = "unknown_pipeline"
        suggested_extractor = "none"

    # 5. Perform dynamic content-based classification if content is available and supported
    if is_supported:
        if primary_category == "image":
            image_bytes = None
            if file_bytes:
                image_bytes = file_bytes
            elif file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        image_bytes = f.read()
                except Exception as e:
                    logger.error(f"Failed to read image file from path {file_path}: {e}")

            if image_bytes:
                from RAG.services.image_classifier import classify_image
                try:
                    class_res = classify_image(image_bytes, mime_type)
                    img_type = class_res.get("image_type", "UNKNOWN")
                    suggested_extractor = IMAGE_EXTRACTOR_MAP.get(img_type, "describe_image_with_gemini")
                    classification = {
                        "type": img_type,
                        "confidence": class_res.get("confidence", 0.0),
                        "reason": class_res.get("reason", "")
                    }
                except Exception as e:
                    logger.error(f"Dynamic image classification failed: {e}")

        elif primary_category == "spreadsheet":
            temp_file_path = None
            if file_path and os.path.exists(file_path):
                temp_file_path = file_path
            elif file_bytes:
                suffix = f".{ext}" if ext else ""
                try:
                    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
                    with os.fdopen(fd, "wb") as f:
                        f.write(file_bytes)
                except Exception as e:
                    logger.error(f"Failed to write temporary file for table classification: {e}")
                    temp_file_path = None

            if temp_file_path:
                from RAG.routes.validation import parse_table_file_to_markdown
                from RAG.services.table_classifier import classify_table
                try:
                    tbl_name = file_name or os.path.basename(temp_file_path)
                    table_markdown = parse_table_file_to_markdown(temp_file_path, tbl_name)
                    class_res = classify_table(table_markdown)
                    tbl_type = class_res.get("table_type", "TABLE_UNKNOWN")
                    suggested_extractor = TABLE_EXTRACTOR_MAP.get(tbl_type, "extract_unknown_table")
                    classification = {
                        "type": tbl_type,
                        "confidence": class_res.get("confidence", 0.0),
                        "reason": class_res.get("reason", "")
                    }
                except Exception as e:
                    logger.error(f"Dynamic table classification failed: {e}")
                finally:
                    # Clean up the temp file if created from bytes
                    if not file_path and temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as re:
                            logger.warning(f"Failed to remove temp file {temp_file_path}: {re}")

    return RoutingMetadata(
        file_name=file_name,
        file_extension=ext,
        mime_type=mime_type,
        primary_category=primary_category,
        is_supported=is_supported,
        suggested_route=suggested_route,
        suggested_extractor=suggested_extractor,
        classification=classification
    )
