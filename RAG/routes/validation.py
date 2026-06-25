import os
import time
import uuid
import shutil
import logging
import tempfile
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pdfplumber
from pypdf import PdfReader

# Imports from RAG modules
from RAG.ingestion import describe_image_with_gemini, table_to_markdown
from RAG.services.table_classifier import classify_table, TableType
from RAG.services.table_intelligence import (
    extract_simple_table,
    extract_comparison_table,
    extract_financial_table,
    extract_statistical_table,
    extract_timeseries_table,
    extract_unknown_table,
    SimpleTableKnowledge,
    ComparisonTableKnowledge,
    FinancialTableKnowledge,
    StatisticalTableKnowledge,
    TimeSeriesTableKnowledge,
    UnknownTableKnowledge
)

# Setup logger
logger = logging.getLogger(__name__)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_VALIDATION_IMAGES_DIR = os.path.join(BASE_DIR, "data", "temp", "validation_images")
os.makedirs(TEMP_VALIDATION_IMAGES_DIR, exist_ok=True)

router = APIRouter()

# --- Response Schemas ---

class ImageSummaryResponse(BaseModel):
    filename: str
    summary: str
    processing_time: Optional[float] = Field(None, description="Time taken to process request in seconds")
    status: Optional[str] = Field(None, description="Status of the operation")

class TableItem(BaseModel):
    page: int
    markdown: str

class ExcelSheetInfo(BaseModel):
    sheet_name: str
    rows: int
    columns: int
    column_names: List[str]

class TabularAnalysisResponse(BaseModel):
    detected_file_type: str
    tables_found: Optional[int] = None
    tables: Optional[List[TableItem]] = None
    sheets: Optional[List[ExcelSheetInfo]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    preview: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None
    file_size_kb: Optional[float] = None
    status: Optional[str] = None

class TextKnowledgeResponse(BaseModel):
    image_type: str = Field("text_image")
    extracted_text: str
    word_count: int
    rich_text_representation: str
    ocr_engine_used: str
    ocr_available: bool
    ocr_raw_text: str
    ocr_text_length: int
    ocr_blocks_detected: int
    average_confidence: float
    cleaned_text: str

class ChartDataPointResponse(BaseModel):
    label: str
    value: str

class ChartKnowledgeResponse(BaseModel):
    image_type: str = Field("chart")
    chart_type: str
    x_axis: str
    y_axis: str
    data_points: List[ChartDataPointResponse]
    insights: List[str]
    rich_text_representation: str

class DiagramEdgeResponse(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    label_or_relationship: Optional[str] = None
    
    class Config:
        populate_by_name = True

class DiagramKnowledgeResponse(BaseModel):
    image_type: str = Field("diagram")
    diagram_type: str
    nodes: List[str]
    edges: List[DiagramEdgeResponse]
    description: str
    rich_text_representation: str

# --- New Validation Schemas ---

class TextImageKnowledge(BaseModel):
    document_type: str = Field(..., description="The type of document identified, e.g., article, report, invoice, code_snippet.")
    extracted_text: str = Field(..., description="The raw extracted OCR text.")
    cleaned_text: str = Field(..., description="The Gemini-cleaned version of the OCR text.")
    key_points: List[str] = Field(..., description="Main points extracted from the text.")
    entities: List[str] = Field(..., description="Key entities mentioned in the text.")
    word_count: int = Field(..., description="Total word count.")

class ChartImageKnowledge(BaseModel):
    chart_type: str = Field(..., description="The classified type of the chart (e.g., bar_chart, line_graph).")
    title: str = Field(..., description="Extracted title of the chart.")
    x_axis: str = Field(..., description="Description or label of the X-axis.")
    y_axis: str = Field(..., description="Description or label of the Y-axis.")
    data_points: List[Dict[str, Any]] = Field(..., description="Extracted numeric coordinates and category labels.")
    trends: List[str] = Field(..., description="Core trends observed in the chart data.")
    insights: List[str] = Field(..., description="Deeper takeaways, peaks, dips, or anomalies.")

class DiagramEdgeModel(BaseModel):
    from_node: str = Field(..., alias="from", description="Source component name.")
    to_node: str = Field(..., alias="to", description="Destination component name.")
    label_or_relationship: Optional[str] = Field(None, description="Relationship description or trigger label.")
    
    class Config:
        populate_by_name = True

class DiagramImageKnowledge(BaseModel):
    nodes: List[str] = Field(..., description="List of components or processes in the diagram.")
    relationships: List[DiagramEdgeModel] = Field(..., description="Connections/edges representing interactions.")
    workflow: List[str] = Field(..., description="Process flow path sequence.")
    components: List[str] = Field(..., description="Identified subsystems/components.")
    summary: str = Field(..., description="Text description of the diagram.")

class MixedImageKnowledgeModel(BaseModel):
    sections: List[str] = Field(..., description="Layout blocks identified in the infographic.")
    headings: List[str] = Field(..., description="Extracted headers and titles.")
    labels: List[str] = Field(..., description="Annotations and callouts.")
    process_flow: List[str] = Field(..., description="Sequential steps in the infographic.")
    key_takeaways: List[str] = Field(..., description="Core takeaways.")
    summary: str = Field(..., description="Comprehensive visual summary.")

class NaturalImageKnowledgeModel(BaseModel):
    scene_type: str = Field(..., description="Type of scene, e.g. indoor, outdoor.")
    objects: List[str] = Field(..., description="Key physical elements/objects.")
    environment: str = Field(..., description="Contextual environment.")
    activities: List[str] = Field(..., description="Actions/activities taking place.")
    summary: str = Field(..., description="Summary of the photograph.")

class UnknownImageKnowledgeModel(BaseModel):
    summary: str = Field(..., description="Fallback visual summary description.")

class ImageValidationResponse(BaseModel):
    image_type: str = Field(..., description="The classified category of the image (TEXT_IMAGE, CHART, DIAGRAM, NATURAL_IMAGE, MIXED, UNKNOWN)")
    confidence: float = Field(..., description="The confidence score of the classification (0.0 to 1.0)")
    reason: str = Field(..., description="Reasoning behind classification")
    extractor_selected: str = Field(..., description="Name of the extractor function utilized")
    knowledge: Union[
        TextImageKnowledge,
        ChartImageKnowledge,
        DiagramImageKnowledge,
        MixedImageKnowledgeModel,
        NaturalImageKnowledgeModel,
        UnknownImageKnowledgeModel
    ] = Field(..., description="Category-specific extracted structured data object")
    rich_text_representation: str = Field(..., description="Formatted markdown block representing the extracted data")
    timings: Dict[str, float] = Field(..., description="Durations of classification, extraction, and total time in seconds")

class PDFImageItem(BaseModel):
    page: int = Field(..., description="1-based page number")
    image_index: int = Field(..., description="1-based index on the page")
    image_type: str = Field(..., description="Classified category")
    extractor: str = Field(..., description="Extractor name")
    knowledge: Dict[str, Any] = Field(..., description="Structured knowledge object extracted")

class PDFImagesValidationResponse(BaseModel):
    total_images: int = Field(..., description="Total count of images extracted and processed")
    images: List[PDFImageItem] = Field(..., description="List of all extracted images and their results")

class ImageClassificationResponse(BaseModel):
    image_type: str = Field(..., description="The classified category")
    confidence: float = Field(..., description="Confidence score")
    reason: str = Field(..., description="Reasoning rationale")

class ImageDebugResponse(BaseModel):
    classification: Dict[str, Any] = Field(..., description="Raw output from classify_image")
    selected_route: str = Field(..., description="Route selected by pipeline")
    raw_extractor_output: Dict[str, Any] = Field(..., description="Raw dictionary returned by extractor")
    rich_text_representation: str = Field(..., description="Generated markdown representation")
    timings: Dict[str, float] = Field(..., description="Phase timings")


# --- Helper Functions ---

def validate_image_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{ext}'. Supported formats: jpg, jpeg, png, webp."
        )


# --- Endpoints ---

@router.post("/image-summary", response_model=ImageSummaryResponse)
async def validate_image_summary(file: UploadFile = File(...), debug: bool = False):
    """Upload a single image and generate a Gemini Vision summary."""
    validate_image_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured."
        )
        
    start_time = time.time()
    try:
        # Save to a temporary file to read bytes & detect MIME
        ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"temp_{uuid.uuid4()}{ext}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
            
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        mime_type = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
            
        summary = describe_image_with_gemini(img_bytes, mime_type)
        if summary == "[Visual element failed to process]":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini summary generation failed."
            )
            
        elapsed = time.time() - start_time
        return ImageSummaryResponse(
            filename=file.filename,
            summary=summary,
            processing_time=round(elapsed, 2) if debug else None,
            status="success" if debug else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image-summary endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )


@router.post("/tabular-data-analysis", response_model=TabularAnalysisResponse)
async def validate_tabular_data_analysis(file: UploadFile = File(...), debug: bool = False):
    """Upload a PDF, Excel spreadsheet (.xlsx, .xls), or CSV file, and analyze/preview the tabular data."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )
        
    start_time = time.time()
    filename = file.filename.lower()
    
    # Detect file type
    if filename.endswith(".pdf"):
        detected_type = "pdf"
    elif filename.endswith(".xlsx"):
        detected_type = "xlsx"
    elif filename.endswith(".xls"):
        detected_type = "xls"
    elif filename.endswith(".csv"):
        detected_type = "csv"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported formats: PDF, XLSX, XLS, CSV."
        )
        
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"tab_temp_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size_kb = round(os.path.getsize(temp_path) / 1024, 2)
        if file_size_kb == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded."
            )
            
        tables_found = None
        tables = None
        sheets = None
        rows = None
        columns = None
        column_names = None
        preview = None
        
        if detected_type == "pdf":
            # PDF Table Extraction Flow
            tables_found_list = []
            preview_data = []
            try:
                with pdfplumber.open(temp_path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        page_num = page_idx + 1
                        extracted_tables = page.extract_tables()
                        for t in extracted_tables:
                            md_table = table_to_markdown(t)
                            if md_table:
                                tables_found_list.append(TableItem(
                                    page=page_num,
                                    markdown=md_table
                                ))
                                # Convert first 5 rows (excluding headers) to structured preview dictionaries
                                if not preview_data and len(t) > 1:
                                    headers = [str(cell or f"Col{i}") for i, cell in enumerate(t[0])]
                                    for row in t[1:6]:
                                        row_dict = {}
                                        for i, cell in enumerate(row):
                                            col_name = headers[i] if i < len(headers) else f"Col{i}"
                                            row_dict[col_name] = str(cell or "")
                                        preview_data.append(row_dict)
            except Exception as pdf_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"The PDF could not be processed because it appears to be corrupted: {str(pdf_err)}"
                )
            tables_found = len(tables_found_list)
            tables = tables_found_list
            preview = preview_data
            
        elif detected_type in ["xlsx", "xls"]:
            # Excel Extraction Flow
            try:
                import pandas as pd
                sheets_list = []
                preview_data = []
                
                with pd.ExcelFile(temp_path) as excel_file:
                    for sheet_name in excel_file.sheet_names:
                        df = excel_file.parse(sheet_name)
                        # Convert NaNs to None for clean JSON serialization
                        df = df.where(pd.notnull(df), None)
                        
                        sheets_list.append(ExcelSheetInfo(
                            sheet_name=sheet_name,
                            rows=len(df),
                            columns=len(df.columns),
                            column_names=[str(c) for c in df.columns]
                        ))
                        
                        # Generate preview from the first sheet's first 5 rows
                        if not preview_data:
                            preview_data = df.head(5).to_dict(orient="records")
                            
                sheets = sheets_list
                preview = preview_data
            except Exception as xl_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupted or invalid Excel file: {str(xl_err)}"
                )
                
        elif detected_type == "csv":
            # CSV Extraction Flow
            try:
                import pandas as pd
                df = pd.read_csv(temp_path)
                df = df.where(pd.notnull(df), None)
                rows = len(df)
                columns = len(df.columns)
                column_names = [str(c) for c in df.columns]
                preview = df.head(5).to_dict(orient="records")
            except Exception as csv_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupted or invalid CSV file: {str(csv_err)}"
                )
                
        elapsed = time.time() - start_time
        return TabularAnalysisResponse(
            detected_file_type=detected_type,
            tables_found=tables_found,
            tables=tables,
            sheets=sheets,
            rows=rows,
            columns=columns,
            column_names=column_names,
            preview=preview,
            processing_time=round(elapsed, 2) if debug else None,
            file_size_kb=file_size_kb if debug else None,
            status="success" if debug else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in tabular-data-analysis endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tabular analysis failed: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/text-knowledge-extraction", response_model=TextKnowledgeResponse)
async def validate_text_knowledge_extraction(file: UploadFile = File(...)):
    """Upload a text-heavy image and extract all text content using OCR and Gemini cleaning."""
    validate_image_file(file)
    ext = os.path.splitext(file.filename)[1].lower()
    mime_type = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".webp":
        mime_type = "image/webp"
        
    try:
        image_bytes = await file.read()
        from RAG.services.text_knowledge_extractor import extract_text_knowledge
        res = extract_text_knowledge(image_bytes, mime_type)
        return res
    except Exception as e:
        logger.error(f"Error in text-knowledge-extraction: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": str(e)
            }
        )


@router.post("/chart-knowledge-extraction", response_model=ChartKnowledgeResponse)
async def validate_chart_knowledge_extraction(file: UploadFile = File(...)):
    """Upload a chart/graph image and extract precise numerical data coordinates and insights."""
    validate_image_file(file)
    ext = os.path.splitext(file.filename)[1].lower()
    mime_type = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".webp":
        mime_type = "image/webp"
        
    try:
        image_bytes = await file.read()
        from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
        res = extract_chart_knowledge(image_bytes, mime_type)
        return res
    except Exception as e:
        logger.error(f"Error in chart-knowledge-extraction: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": str(e)
            }
        )


@router.post("/diagram-knowledge-extraction", response_model=DiagramKnowledgeResponse)
async def validate_diagram_knowledge_extraction(file: UploadFile = File(...)):
    """Upload a flowchart or architecture diagram image and extract topology (nodes, edges, descriptions)."""
    validate_image_file(file)
    ext = os.path.splitext(file.filename)[1].lower()
    mime_type = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".webp":
        mime_type = "image/webp"
        
    try:
        image_bytes = await file.read()
        from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
        res = extract_diagram_knowledge(image_bytes, mime_type)
        # Map output edges structure
        res["edges"] = [{"from": e["from"], "to": e["to"], "label_or_relationship": e["label_or_relationship"]} for e in res["edges"]]
        return res
    except Exception as e:
        logger.error(f"Error in diagram-knowledge-extraction: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": str(e)
            }
        )

def run_visual_understanding_logic_on_bytes(image_bytes: bytes, mime_type: str) -> dict:
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

@router.post("/image", response_model=ImageValidationResponse, description="Upload a single image and run classification, routing, and specialized extraction. Returns timing and structured schema.")
async def validate_image(file: UploadFile = File(...)):
    validate_image_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_total_start = time.perf_counter()
    image_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    # 1. Classification Phase
    t_class_start = time.perf_counter()
    from RAG.services.image_classifier import classify_image
    try:
        classifier_result = classify_image(image_bytes, mime)
        image_type = classifier_result.get("image_type", "UNKNOWN")
        confidence = classifier_result.get("confidence", 0.0)
        reason = classifier_result.get("reason", "")
    except Exception as class_err:
        logger.error(f"Image classification failed: {class_err}")
        image_type = "UNKNOWN"
        confidence = 0.0
        reason = f"Classification exception: {class_err}"
    t_class_end = time.perf_counter()
    class_time = round(t_class_end - t_class_start, 4)

    # 2. Extraction & Routing Phase
    t_extract_start = time.perf_counter()
    extractor_selected = ""
    knowledge = {}
    rich_text = ""
    success = False

    try:
        if image_type == "TEXT_IMAGE":
            extractor_selected = "extract_text_knowledge"
            from RAG.services.text_knowledge_extractor import extract_text_knowledge
            res = extract_text_knowledge(image_bytes, mime)
            knowledge = {
                "document_type": res.get("document_type", "unspecified"),
                "extracted_text": res.get("extracted_text", ""),
                "cleaned_text": res.get("cleaned_text", ""),
                "key_points": res.get("key_points", []),
                "entities": res.get("entities", []),
                "word_count": res.get("word_count", 0)
            }
            rich_text = res.get("rich_text_representation", "")
            success = True
        elif image_type == "CHART":
            extractor_selected = "extract_chart_knowledge"
            from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
            res = extract_chart_knowledge(image_bytes, mime)
            knowledge = {
                "chart_type": res.get("chart_type", ""),
                "title": res.get("title", ""),
                "x_axis": res.get("x_axis", ""),
                "y_axis": res.get("y_axis", ""),
                "data_points": res.get("data_points", []),
                "trends": res.get("trends", []),
                "insights": res.get("insights", [])
            }
            rich_text = res.get("rich_text_representation", "")
            success = True
        elif image_type == "DIAGRAM":
            extractor_selected = "extract_diagram_knowledge"
            from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
            res = extract_diagram_knowledge(image_bytes, mime)
            relationships = []
            for edge in res.get("relationships", []):
                relationships.append({
                    "from": edge.get("from") or edge.get("from_node") or "",
                    "to": edge.get("to") or edge.get("to_node") or "",
                    "label_or_relationship": edge.get("label_or_relationship")
                })
            knowledge = {
                "nodes": res.get("nodes", []),
                "relationships": relationships,
                "workflow": res.get("workflow", []),
                "components": res.get("components", []),
                "summary": res.get("summary", "")
            }
            rich_text = res.get("rich_text_representation", "")
            success = True
        elif image_type == "MIXED":
            extractor_selected = "run_visual_understanding_logic"
            res = run_visual_understanding_logic_on_bytes(image_bytes, mime)
            knowledge = {
                "sections": res.get("sections", []),
                "headings": res.get("headings", []),
                "labels": res.get("labels", []),
                "process_flow": res.get("process_flow", []),
                "key_takeaways": res.get("key_takeaways", []),
                "summary": res.get("summary", "")
            }
            rich_text = res.get("rich_text_representation", "")
            success = True
        elif image_type == "NATURAL_IMAGE":
            extractor_selected = "extract_natural_image_knowledge"
            from RAG.services.image_intelligence import extract_natural_image_knowledge
            res = extract_natural_image_knowledge(image_bytes, mime)
            knowledge = {
                "scene_type": res.get("scene_type", ""),
                "objects": res.get("objects", []),
                "environment": res.get("environment", ""),
                "activities": res.get("activities", []),
                "summary": res.get("summary", "")
            }
            rich_text = res.get("rich_text_representation", "")
            success = True
        else:
            raise ValueError(f"Unknown or low-confidence type: {image_type}")

    except Exception as ext_err:
        logger.error(f"Extractor {extractor_selected} failed for image type {image_type}: {ext_err}. Falling back to description.")
        try:
            extractor_selected = f"describe_image_with_gemini (fallback after {extractor_selected} failure)"
            desc = describe_image_with_gemini(image_bytes, mime)
            knowledge = {
                "summary": desc
            }
            rich_text = f"# Image Description (Fallback after error)\n\n## Gemini Vision Summary:\n{desc}\n"
            success = True
        except Exception as fallback_err:
            logger.critical(f"Fallback extractor describe_image_with_gemini also failed: {fallback_err}")
            knowledge = {
                "summary": f"Failed to extract knowledge: {ext_err}"
            }
            rich_text = f"# Extraction Failed\n\nError: {ext_err}"

    t_extract_end = time.perf_counter()
    extract_time = round(t_extract_end - t_extract_start, 4)
    total_time = round(time.perf_counter() - t_total_start, 4)

    timings = {
        "classification_seconds": class_time,
        "extraction_seconds": extract_time,
        "total_seconds": total_time
    }

    # Detailed logging
    logger.info(
        "Multimodal validation | file=%s | classifier_result=%s | confidence=%.4f | extractor=%s | extraction_time=%.4fs | output_size=%d chars | success=%s",
        file.filename, image_type, confidence, extractor_selected, extract_time, len(rich_text), success
    )

    return ImageValidationResponse(
        image_type=image_type,
        confidence=confidence,
        reason=reason,
        extractor_selected=extractor_selected,
        knowledge=knowledge,
        rich_text_representation=rich_text,
        timings=timings
    )

@router.post("/pdf-images", response_model=PDFImagesValidationResponse, description="Upload a PDF and run the extraction, classification, and routing extractors sequentially on all embedded images.")
async def validate_pdf_images(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF."
        )
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_start = time.perf_counter()
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"pdf_temp_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pypdf_reader = PdfReader(temp_path)
        total_pages = len(pypdf_reader.pages)
        processed_images = []
        image_idx_counter = 1

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            pypdf_page = pypdf_reader.pages[page_idx]
            
            if not hasattr(pypdf_page, "images") or not pypdf_page.images:
                continue

            for img_idx, img in enumerate(pypdf_page.images, 1):
                mime = "image/png"
                if img.name.lower().endswith(".jpg") or img.name.lower().endswith(".jpeg"):
                    mime = "image/jpeg"
                elif img.name.lower().endswith(".webp"):
                    mime = "image/webp"

                # 1. Classify
                from RAG.services.image_classifier import classify_image
                try:
                    classifier_result = classify_image(img.data, mime)
                    image_type = classifier_result.get("image_type", "UNKNOWN")
                except Exception as ce:
                    logger.error(f"Classification failed for PDF image (p.{page_num}, img.{img_idx}): {ce}")
                    image_type = "UNKNOWN"

                # 2. Extract
                extractor_selected = ""
                knowledge = {}
                try:
                    if image_type == "TEXT_IMAGE":
                        extractor_selected = "extract_text_knowledge"
                        from RAG.services.text_knowledge_extractor import extract_text_knowledge
                        res = extract_text_knowledge(img.data, mime)
                        knowledge = {
                            "document_type": res.get("document_type", "unspecified"),
                            "extracted_text": res.get("extracted_text", ""),
                            "cleaned_text": res.get("cleaned_text", ""),
                            "key_points": res.get("key_points", []),
                            "entities": res.get("entities", []),
                            "word_count": res.get("word_count", 0)
                        }
                    elif image_type == "CHART":
                        extractor_selected = "extract_chart_knowledge"
                        from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
                        res = extract_chart_knowledge(img.data, mime)
                        knowledge = {
                            "chart_type": res.get("chart_type", ""),
                            "title": res.get("title", ""),
                            "x_axis": res.get("x_axis", ""),
                            "y_axis": res.get("y_axis", ""),
                            "data_points": res.get("data_points", []),
                            "trends": res.get("trends", []),
                            "insights": res.get("insights", [])
                        }
                    elif image_type == "DIAGRAM":
                        extractor_selected = "extract_diagram_knowledge"
                        from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
                        res = extract_diagram_knowledge(img.data, mime)
                        relationships = []
                        for edge in res.get("relationships", []):
                            relationships.append({
                                "from": edge.get("from") or edge.get("from_node") or "",
                                "to": edge.get("to") or edge.get("to_node") or "",
                                "label_or_relationship": edge.get("label_or_relationship")
                            })
                        knowledge = {
                            "nodes": res.get("nodes", []),
                            "relationships": relationships,
                            "workflow": res.get("workflow", []),
                            "components": res.get("components", []),
                            "summary": res.get("summary", "")
                        }
                    elif image_type == "MIXED":
                        extractor_selected = "run_visual_understanding_logic"
                        res = run_visual_understanding_logic_on_bytes(img.data, mime)
                        knowledge = {
                            "sections": res.get("sections", []),
                            "headings": res.get("headings", []),
                            "labels": res.get("labels", []),
                            "process_flow": res.get("process_flow", []),
                            "key_takeaways": res.get("key_takeaways", []),
                            "summary": res.get("summary", "")
                        }
                    elif image_type == "NATURAL_IMAGE":
                        extractor_selected = "extract_natural_image_knowledge"
                        from RAG.services.image_intelligence import extract_natural_image_knowledge
                        res = extract_natural_image_knowledge(img.data, mime)
                        knowledge = {
                            "scene_type": res.get("scene_type", ""),
                            "objects": res.get("objects", []),
                            "environment": res.get("environment", ""),
                            "activities": res.get("activities", []),
                            "summary": res.get("summary", "")
                        }
                    else:
                        raise ValueError(f"Unknown type: {image_type}")

                except Exception as ext_err:
                    logger.error(f"PDF extraction failed for p.{page_num}, img.{img_idx} (extractor: {extractor_selected}): {ext_err}")
                    try:
                        extractor_selected = f"describe_image_with_gemini (fallback)"
                        desc = describe_image_with_gemini(img.data, mime)
                        knowledge = {"summary": desc}
                    except Exception as fallback_err:
                        extractor_selected = "none"
                        knowledge = {"summary": f"Failed: {ext_err}"}

                processed_images.append(PDFImageItem(
                    page=page_num,
                    image_index=image_idx_counter,
                    image_type=image_type,
                    extractor=extractor_selected,
                    knowledge=knowledge
                ))
                image_idx_counter += 1

        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(f"Processed PDF images | total_images={len(processed_images)} | time={elapsed}s")

        return PDFImagesValidationResponse(
            total_images=len(processed_images),
            images=processed_images
        )
    except Exception as e:
        logger.error(f"Failed to process PDF images: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF images: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/image-classify", response_model=ImageClassificationResponse, description="Perform image classification only, returning category type, confidence, and visual reasoning.")
async def validate_image_classify(file: UploadFile = File(...)):
    validate_image_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_start = time.perf_counter()
    image_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    from RAG.services.image_classifier import classify_image
    try:
        result = classify_image(image_bytes, mime)
        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(f"Classified image | type={result.get('image_type')} | time={elapsed}s")
        return ImageClassificationResponse(
            image_type=result.get("image_type", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            reason=result.get("reason", "")
        )
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )

@router.post("/image-debug", response_model=ImageDebugResponse, description="Developer debugging endpoint returning raw classification dictionary, selected route string, raw extractor output dictionary, rich text, and timings.")
async def validate_image_debug(file: UploadFile = File(...)):
    validate_image_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_total_start = time.perf_counter()
    image_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    # 1. Classification
    t_class_start = time.perf_counter()
    from RAG.services.image_classifier import classify_image
    try:
        classifier_result = classify_image(image_bytes, mime)
    except Exception as ce:
        classifier_result = {"image_type": "UNKNOWN", "confidence": 0.0, "reason": f"Failed: {ce}"}
    class_time = round(time.perf_counter() - t_class_start, 4)

    image_type = classifier_result.get("image_type", "UNKNOWN")

    # 2. Extraction
    t_extract_start = time.perf_counter()
    extractor_selected = ""
    raw_extractor_output = {}
    rich_text = ""

    try:
        if image_type == "TEXT_IMAGE":
            extractor_selected = "extract_text_knowledge"
            from RAG.services.text_knowledge_extractor import extract_text_knowledge
            raw_extractor_output = extract_text_knowledge(image_bytes, mime)
            rich_text = raw_extractor_output.get("rich_text_representation", "")
        elif image_type == "CHART":
            extractor_selected = "extract_chart_knowledge"
            from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
            raw_extractor_output = extract_chart_knowledge(image_bytes, mime)
            rich_text = raw_extractor_output.get("rich_text_representation", "")
        elif image_type == "DIAGRAM":
            extractor_selected = "extract_diagram_knowledge"
            from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
            raw_extractor_output = extract_diagram_knowledge(image_bytes, mime)
            rich_text = raw_extractor_output.get("rich_text_representation", "")
        elif image_type == "MIXED":
            extractor_selected = "run_visual_understanding_logic"
            raw_extractor_output = run_visual_understanding_logic_on_bytes(image_bytes, mime)
            rich_text = raw_extractor_output.get("rich_text_representation", "")
        elif image_type == "NATURAL_IMAGE":
            extractor_selected = "extract_natural_image_knowledge"
            from RAG.services.image_intelligence import extract_natural_image_knowledge
            raw_extractor_output = extract_natural_image_knowledge(image_bytes, mime)
            rich_text = raw_extractor_output.get("rich_text_representation", "")
        else:
            extractor_selected = "describe_image_with_gemini"
            desc = describe_image_with_gemini(image_bytes, mime)
            raw_extractor_output = {"description": desc}
            rich_text = f"# Legacy Description\n\n{desc}\n"

    except Exception as ext_err:
        logger.error(f"Debug extractor failed: {ext_err}")
        extractor_selected = f"describe_image_with_gemini (fallback after {extractor_selected} failure)"
        try:
            desc = describe_image_with_gemini(image_bytes, mime)
            raw_extractor_output = {"description": desc, "error": str(ext_err)}
            rich_text = f"# Legacy Description (Fallback)\n\n{desc}\n"
        except Exception as fb_err:
            raw_extractor_output = {"error": str(ext_err), "fallback_error": str(fb_err)}
            rich_text = f"# Extraction and Fallback Failed\n\nError: {ext_err}"

    extract_time = round(time.perf_counter() - t_extract_start, 4)
    total_time = round(time.perf_counter() - t_total_start, 4)

    timings = {
        "classification_seconds": class_time,
        "extraction_seconds": extract_time,
        "total_seconds": total_time
    }

    return ImageDebugResponse(
        classification=classifier_result,
        selected_route=extractor_selected,
        raw_extractor_output=raw_extractor_output,
        rich_text_representation=rich_text,
        timings=timings
    )

# --- Table Validation Response Schemas ---

class TableClassificationResponse(BaseModel):
    table_type: str = Field(..., description="The classified category of the table")
    confidence: float = Field(..., description="Classification confidence score")
    reason: str = Field(..., description="Reasoning narrative")

class TableValidationResponse(BaseModel):
    table_type: str = Field(..., description="Classified category of the table")
    confidence: float = Field(..., description="Classification confidence score")
    reason: str = Field(..., description="Reasoning narrative")
    extractor_selected: str = Field(..., description="The name of the specialized extractor invoked")
    knowledge: Union[
        SimpleTableKnowledge,
        ComparisonTableKnowledge,
        FinancialTableKnowledge,
        StatisticalTableKnowledge,
        TimeSeriesTableKnowledge,
        UnknownTableKnowledge
    ] = Field(..., description="Category-specific extracted structured data object")
    rich_text_representation: str = Field(..., description="Formatted Markdown block representation")
    timings: Dict[str, float] = Field(..., description="Durations of classification, extraction, and total time in seconds")

class PDFTableItem(BaseModel):
    page: int = Field(..., description="1-based page number")
    table_index: int = Field(..., description="1-based index on the page")
    table_type: str = Field(..., description="Classified category")
    extractor: str = Field(..., description="Extractor name")
    knowledge: Dict[str, Any] = Field(..., description="Structured knowledge object extracted")

class PDFTablesValidationResponse(BaseModel):
    total_tables: int = Field(..., description="Total count of tables extracted and processed")
    tables: List[PDFTableItem] = Field(..., description="List of all extracted tables and their results")

class TableDebugResponse(BaseModel):
    classification: Dict[str, Any] = Field(..., description="Raw output from classify_table")
    selected_route: str = Field(..., description="Route selected by pipeline")
    raw_extractor_output: Dict[str, Any] = Field(..., description="Raw dictionary returned by extractor")
    rich_text_representation: str = Field(..., description="Generated markdown representation")
    timings: Dict[str, float] = Field(..., description="Phase timings")

# --- Table Processing Helpers ---

def parse_table_file_to_markdown(temp_path: str, filename: str) -> str:
    """Reads a file (CSV, Excel, PDF, or text/markdown) and extracts/converts it to a markdown table string."""
    import pandas as pd
    lower_name = filename.lower()
    
    if lower_name.endswith(".csv"):
        df = pd.read_csv(temp_path)
        df = df.where(pd.notnull(df), "")
        cols = [str(c) for c in df.columns]
        rows = [[str(val) for val in row] for row in df.values]
        return table_to_markdown([cols] + rows)
        
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(temp_path)
        df = df.where(pd.notnull(df), "")
        cols = [str(c) for c in df.columns]
        rows = [[str(val) for val in row] for row in df.values]
        return table_to_markdown([cols] + rows)
        
    elif lower_name.endswith(".pdf"):
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for t in tables:
                    md = table_to_markdown(t)
                    if md:
                        return md
        raise ValueError("No tabular data detected in PDF file.")
        
    elif lower_name.endswith((".md", ".txt")):
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError("Text/Markdown table file is empty.")
        return content
        
    else:
        raise ValueError(f"Unsupported table format: {os.path.splitext(filename)[1]}")

def validate_table_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls", ".pdf", ".md", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Supported table formats: csv, xlsx, xls, pdf, md, txt."
        )

# --- Table validation endpoints ---

@router.post("/table-classify", response_model=TableClassificationResponse, description="Perform table classification only, returning category type, confidence, and reasoning.")
async def validate_table_classify(file: UploadFile = File(...)):
    validate_table_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_start = time.perf_counter()
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"tbl_cls_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        table_markdown = parse_table_file_to_markdown(temp_path, file.filename)
        classifier_result = classify_table(table_markdown)
        elapsed = round(time.perf_counter() - t_start, 4)
        
        logger.info(f"Classified table | type={classifier_result.get('table_type')} | time={elapsed}s")
        return TableClassificationResponse(
            table_type=classifier_result.get("table_type", "TABLE_UNKNOWN"),
            confidence=classifier_result.get("confidence", 0.0),
            reason=classifier_result.get("reason", "")
        )
    except Exception as e:
        logger.error(f"Table classification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/table", response_model=TableValidationResponse, description="Upload a single table file, extract/parse it, classify, route to specialized extractor, and return structured output and timings.")
async def validate_table(file: UploadFile = File(...)):
    validate_table_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_total_start = time.perf_counter()
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"tbl_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        table_markdown = parse_table_file_to_markdown(temp_path, file.filename)
        
        # 1. Classification
        t_class_start = time.perf_counter()
        classifier_result = classify_table(table_markdown)
        table_type = classifier_result.get("table_type", "TABLE_UNKNOWN")
        confidence = classifier_result.get("confidence", 0.0)
        reason = classifier_result.get("reason", "")
        t_class_end = time.perf_counter()
        class_time = round(t_class_end - t_class_start, 4)
        
        # 2. Ingestion routing
        t_extract_start = time.perf_counter()
        extractor_selected = ""
        knowledge = {}
        rich_text = ""
        success = False
        
        try:
            if table_type == "TABLE_SIMPLE":
                extractor_selected = "extract_simple_table"
                res = extract_simple_table(table_markdown)
                knowledge = {
                    "title": res.get("title"),
                    "headers": res.get("headers", []),
                    "rows": res.get("rows", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            elif table_type == "TABLE_COMPARISON":
                extractor_selected = "extract_comparison_table"
                res = extract_comparison_table(table_markdown)
                knowledge = {
                    "title": res.get("title"),
                    "entities_compared": res.get("entities_compared", []),
                    "attributes_compared": res.get("attributes_compared", []),
                    "comparisons": res.get("comparisons", []),
                    "key_differences": res.get("key_differences", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            elif table_type == "TABLE_FINANCIAL":
                extractor_selected = "extract_financial_table"
                res = extract_financial_table(table_markdown)
                knowledge = {
                    "title": res.get("title"),
                    "reporting_period": res.get("reporting_period"),
                    "currency": res.get("currency"),
                    "financial_metrics": res.get("financial_metrics", []),
                    "key_financial_takeaways": res.get("key_financial_takeaways", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            elif table_type == "TABLE_STATISTICAL":
                extractor_selected = "extract_statistical_table"
                res = extract_statistical_table(table_markdown)
                knowledge = {
                    "title": res.get("title"),
                    "variables": res.get("variables", []),
                    "metrics": res.get("metrics", []),
                    "data_summary": res.get("data_summary", []),
                    "statistical_conclusions": res.get("statistical_conclusions", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            elif table_type == "TABLE_TIMESERIES":
                extractor_selected = "extract_timeseries_table"
                res = extract_timeseries_table(table_markdown)
                knowledge = {
                    "title": res.get("title"),
                    "time_interval": res.get("time_interval", ""),
                    "timestamps": res.get("timestamps", []),
                    "series_data": res.get("series_data", []),
                    "trends_observed": res.get("trends_observed", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            else:
                raise ValueError(f"Unknown or fallback table type: {table_type}")
                
        except Exception as ext_err:
            logger.warning(f"Table extractor {extractor_selected} failed: {ext_err}. Falling back to extract_unknown_table.")
            extractor_selected = "extract_unknown_table (fallback)"
            try:
                res = extract_unknown_table(table_markdown)
                knowledge = {
                    "raw_content_description": res.get("raw_content_description", ""),
                    "key_columns_detected": res.get("key_columns_detected", []),
                    "summary": res.get("summary", "")
                }
                rich_text = res.get("rich_text_representation", "")
                success = True
            except Exception as fallback_err:
                logger.critical(f"Fallback table extractor failed: {fallback_err}")
                knowledge = {
                    "summary": f"Failed: {ext_err}",
                    "raw_content_description": table_markdown,
                    "key_columns_detected": []
                }
                rich_text = f"# Table Extraction Failed\n\nError: {ext_err}\n\n## Content:\n{table_markdown}"
                
        t_extract_end = time.perf_counter()
        extract_time = round(t_extract_end - t_extract_start, 4)
        total_time = round(time.perf_counter() - t_total_start, 4)
        
        timings = {
            "classification_seconds": class_time,
            "extraction_seconds": extract_time,
            "total_seconds": total_time
        }
        
        logger.info(
            "Table validation | file=%s | type=%s | extractor=%s | time=%.4fs | success=%s",
            file.filename, table_type, extractor_selected, total_time, success
        )
        
        return TableValidationResponse(
            table_type=table_type,
            confidence=confidence,
            reason=reason,
            extractor_selected=extractor_selected,
            knowledge=knowledge,
            rich_text_representation=rich_text,
            timings=timings
        )
        
    except Exception as e:
        logger.error(f"Failed to process table validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Table processing failed: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/pdf-tables", response_model=PDFTablesValidationResponse, description="Upload a PDF and sequentially extract, classify, and analyze all contained tables page-by-page.")
async def validate_pdf_tables(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF."
        )
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_start = time.perf_counter()
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"pdf_tbl_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        processed_tables = []
        table_idx_counter = 1
        
        with pdfplumber.open(temp_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables()
                if not tables:
                    continue
                    
                for sub_idx, t in enumerate(tables, 1):
                    table_markdown = table_to_markdown(t)
                    if not table_markdown.strip():
                        continue
                        
                    # 1. Classify
                    try:
                        classifier_result = classify_table(table_markdown)
                        table_type = classifier_result.get("table_type", "TABLE_UNKNOWN")
                    except Exception as ce:
                        logger.error(f"Classification failed for PDF table (p.{page_num}, tbl.{sub_idx}): {ce}")
                        table_type = "TABLE_UNKNOWN"
                        
                    # 2. Extract
                    extractor_selected = ""
                    knowledge = {}
                    try:
                        if table_type == "TABLE_SIMPLE":
                            extractor_selected = "extract_simple_table"
                            res = extract_simple_table(table_markdown)
                            knowledge = {
                                "title": res.get("title"),
                                "headers": res.get("headers", []),
                                "rows": res.get("rows", []),
                                "summary": res.get("summary", "")
                            }
                        elif table_type == "TABLE_COMPARISON":
                            extractor_selected = "extract_comparison_table"
                            res = extract_comparison_table(table_markdown)
                            knowledge = {
                                "title": res.get("title"),
                                "entities_compared": res.get("entities_compared", []),
                                "attributes_compared": res.get("attributes_compared", []),
                                "comparisons": res.get("comparisons", []),
                                "key_differences": res.get("key_differences", []),
                                "summary": res.get("summary", "")
                            }
                        elif table_type == "TABLE_FINANCIAL":
                            extractor_selected = "extract_financial_table"
                            res = extract_financial_table(table_markdown)
                            knowledge = {
                                "title": res.get("title"),
                                "reporting_period": res.get("reporting_period"),
                                "currency": res.get("currency"),
                                "financial_metrics": res.get("financial_metrics", []),
                                "key_financial_takeaways": res.get("key_financial_takeaways", []),
                                "summary": res.get("summary", "")
                            }
                        elif table_type == "TABLE_STATISTICAL":
                            extractor_selected = "extract_statistical_table"
                            res = extract_statistical_table(table_markdown)
                            knowledge = {
                                "title": res.get("title"),
                                "variables": res.get("variables", []),
                                "metrics": res.get("metrics", []),
                                "data_summary": res.get("data_summary", []),
                                "statistical_conclusions": res.get("statistical_conclusions", []),
                                "summary": res.get("summary", "")
                            }
                        elif table_type == "TABLE_TIMESERIES":
                            extractor_selected = "extract_timeseries_table"
                            res = extract_timeseries_table(table_markdown)
                            knowledge = {
                                "title": res.get("title"),
                                "time_interval": res.get("time_interval", ""),
                                "timestamps": res.get("timestamps", []),
                                "series_data": res.get("series_data", []),
                                "trends_observed": res.get("trends_observed", []),
                                "summary": res.get("summary", "")
                            }
                        else:
                            raise ValueError(f"Unknown or fallback table type: {table_type}")
                    except Exception as ext_err:
                        logger.error(f"PDF table extractor failed for p.{page_num}, tbl.{sub_idx}: {ext_err}")
                        extractor_selected = "extract_unknown_table (fallback)"
                        try:
                            res = extract_unknown_table(table_markdown)
                            knowledge = {
                                "raw_content_description": res.get("raw_content_description", ""),
                                "key_columns_detected": res.get("key_columns_detected", []),
                                "summary": res.get("summary", "")
                            }
                        except Exception as fallback_err:
                            extractor_selected = "none"
                            knowledge = {"summary": f"Failed: {ext_err}"}
                            
                    processed_tables.append(PDFTableItem(
                        page=page_num,
                        table_index=table_idx_counter,
                        table_type=table_type,
                        extractor=extractor_selected,
                        knowledge=knowledge
                    ))
                    table_idx_counter += 1
                    
        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(f"Processed PDF tables | total_tables={len(processed_tables)} | time={elapsed}s")
        return PDFTablesValidationResponse(
            total_tables=len(processed_tables),
            tables=processed_tables
        )
        
    except Exception as e:
        logger.error(f"Failed to process PDF tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF tables: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/table-debug", response_model=TableDebugResponse, description="Developer debugging endpoint returning raw classification dictionary, selected route string, raw extractor output dictionary, rich text, and timings.")
async def validate_table_debug(file: UploadFile = File(...)):
    validate_table_file(file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured."
        )

    t_total_start = time.perf_counter()
    temp_path = os.path.join(TEMP_VALIDATION_IMAGES_DIR, f"tbl_dbg_{uuid.uuid4()}_{os.path.basename(file.filename)}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        table_markdown = parse_table_file_to_markdown(temp_path, file.filename)
        
        # 1. Classification
        t_class_start = time.perf_counter()
        classifier_result = classify_table(table_markdown)
        table_type = classifier_result.get("table_type", "TABLE_UNKNOWN")
        class_time = round(time.perf_counter() - t_class_start, 4)
        
        # 2. Extraction
        t_extract_start = time.perf_counter()
        extractor_selected = ""
        raw_extractor_output = {}
        rich_text = ""
        
        try:
            if table_type == "TABLE_SIMPLE":
                extractor_selected = "extract_simple_table"
                raw_extractor_output = extract_simple_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            elif table_type == "TABLE_COMPARISON":
                extractor_selected = "extract_comparison_table"
                raw_extractor_output = extract_comparison_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            elif table_type == "TABLE_FINANCIAL":
                extractor_selected = "extract_financial_table"
                raw_extractor_output = extract_financial_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            elif table_type == "TABLE_STATISTICAL":
                extractor_selected = "extract_statistical_table"
                raw_extractor_output = extract_statistical_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            elif table_type == "TABLE_TIMESERIES":
                extractor_selected = "extract_timeseries_table"
                raw_extractor_output = extract_timeseries_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            else:
                extractor_selected = "extract_unknown_table"
                raw_extractor_output = extract_unknown_table(table_markdown)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
                
        except Exception as ext_err:
            logger.error(f"Debug table extractor failed: {ext_err}")
            extractor_selected = f"extract_unknown_table (fallback after {extractor_selected} failure)"
            try:
                raw_extractor_output = extract_unknown_table(table_markdown)
                raw_extractor_output["error"] = str(ext_err)
                rich_text = raw_extractor_output.get("rich_text_representation", "")
            except Exception as fb_err:
                raw_extractor_output = {"error": str(ext_err), "fallback_error": str(fb_err)}
                rich_text = f"# Extraction and Fallback Failed\n\nError: {ext_err}"
                
        extract_time = round(time.perf_counter() - t_extract_start, 4)
        total_time = round(time.perf_counter() - t_total_start, 4)
        
        timings = {
            "classification_seconds": class_time,
            "extraction_seconds": extract_time,
            "total_seconds": total_time
        }
        
        return TableDebugResponse(
            classification=classifier_result,
            selected_route=extractor_selected,
            raw_extractor_output=raw_extractor_output,
            rich_text_representation=rich_text,
            timings=timings
        )
        
    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug endpoint failed: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
