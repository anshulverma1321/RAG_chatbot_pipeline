import os
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
import pdfplumber
from pypdf import PdfReader

# Core utilities — always needed, no OCR dependency
from RAG.ingestion import split_text, table_to_markdown, describe_image_with_gemini
from RAG.services.image_classifier import classify_image
from RAG.services.table_classifier import classify_table

# Table extractors — Gemini-only, no PaddleOCR dependency
from RAG.services.table_intelligence import (
    extract_simple_table,
    extract_comparison_table,
    extract_financial_table,
    extract_statistical_table,
    extract_timeseries_table,
    extract_unknown_table,
)

# NOTE: Image extractors (chart, diagram, text_knowledge, visual_understanding) are imported
# inside the image routing block at call time to avoid loading PaddleOCR for documents
# that contain no embedded image assets. See the image routing section in orchestrate_pdf().

from RAG.knowledge_normalizer import NormalizedKnowledgeChunk, normalize_knowledge_chunk
from RAG.logger import execution_stage_var, record_performance_timing

logger = logging.getLogger(__name__)


def orchestrate_pdf(file_path: str) -> List[NormalizedKnowledgeChunk]:
    """
    Parses a PDF page-by-page, extracting text, table, and image assets.
    Routes each asset to the appropriate classifier and extractor pipeline,
    and returns a list of NormalizedKnowledgeChunk instances.
    
    Does not interact with database tables, embeddings, or vector stores.
    """
    filename = os.path.basename(file_path)
    logger.info("Starting PDF orchestration with knowledge normalization | file=%s", filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    chunks: List[NormalizedKnowledgeChunk] = []

    try:
        pypdf_reader = PdfReader(file_path)
        total_pages = len(pypdf_reader.pages)
        pdf = pdfplumber.open(file_path)
    except Exception as exc:
        logger.exception("Failed to open or parse PDF %s: %s", filename, exc)
        raise RuntimeError(f"Failed to load PDF file: {exc}")

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        plumber_page = pdf.pages[page_idx]
        pypdf_page = pypdf_reader.pages[page_idx]

        logger.info("PDF page processing start | page=%d | file=%s", page_num, filename)
        logger.info("Orchestrator processing page %d/%d | file=%s", page_num, total_pages, filename)

        # ----------------------------------------------------
        # 1. Table Extraction & Routing
        # ----------------------------------------------------
        stage_token_tbl = execution_stage_var.set("table_pipeline")
        try:
            tables = plumber_page.extract_tables()
            for table_idx, t in enumerate(tables):
                md_table = table_to_markdown(t)
                if not md_table or not md_table.strip():
                    continue

                logger.info("PDF table extraction | page=%d | table_index=%d", page_num, table_idx)

                # Run classifier
                table_type = "TABLE_UNKNOWN"
                confidence = 0.0
                reason = "Default fallback"
                try:
                    import time
                    start_cls = time.time()
                    classifier_result = classify_table(md_table)
                    record_performance_timing("classification_time", time.time() - start_cls)
                    
                    table_type = classifier_result.get("table_type", "TABLE_UNKNOWN")
                    confidence = classifier_result.get("confidence", 0.0)
                    reason = classifier_result.get("reason", "Classification successful")
                except Exception as class_err:
                    logger.exception("Table classification failed on page %d: %s", page_num, class_err)

                logger.info("TABLE PIPELINE classifier selected | type=%s", table_type)

                # Route to specialized extractors
                selected_extractor = ""
                extraction_result = None
                try:
                    if table_type == "TABLE_SIMPLE":
                        selected_extractor = "extract_simple_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_simple_table(md_table)
                    elif table_type == "TABLE_COMPARISON":
                        selected_extractor = "extract_comparison_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_comparison_table(md_table)
                    elif table_type == "TABLE_FINANCIAL":
                        selected_extractor = "extract_financial_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_financial_table(md_table)
                    elif table_type == "TABLE_STATISTICAL":
                        selected_extractor = "extract_statistical_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_statistical_table(md_table)
                    elif table_type == "TABLE_TIMESERIES":
                        selected_extractor = "extract_timeseries_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_timeseries_table(md_table)
                    else:
                        selected_extractor = "extract_unknown_table"
                        logger.info("TABLE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_unknown_table(md_table)
                    logger.info("TABLE PIPELINE extraction completed | extractor=%s", selected_extractor)
                except Exception as ext_err:
                    logger.exception("Table extractor %s failed on page %d: %s", selected_extractor, page_num, ext_err)
                    # Fallback to unknown extractor
                    try:
                        selected_extractor = "extract_unknown_table (fallback)"
                        extraction_result = extract_unknown_table(md_table)
                    except Exception as fallback_err:
                        logger.exception("Fallback table extractor extract_unknown_table also failed: %s", fallback_err)
                        extraction_result = {
                            "title": "Irregular Table",
                            "summary": "Table extraction failed.",
                            "rich_text_representation": md_table
                        }

                # Construct normalized chunk
                rich_text = extraction_result.pop("rich_text_representation", "")
                normalized = normalize_knowledge_chunk(
                    document_name=filename,
                    page_number=page_num,
                    asset_type="table",
                    classification_type=table_type,
                    extractor_used=selected_extractor,
                    knowledge_object=extraction_result,
                    rich_text_representation=rich_text
                )
                chunks.append(normalized)
        except Exception as table_err:
            logger.exception("Failed to extract tables on page %d: %s", page_num, table_err)
        finally:
            execution_stage_var.reset(stage_token_tbl)

        # ----------------------------------------------------
        # 2. Image Extraction & Routing
        # ----------------------------------------------------
        stage_token_img = execution_stage_var.set("image_pipeline")
        try:
            page_images = list(pypdf_page.images) if hasattr(pypdf_page, "images") else []

            if not page_images:
                logger.debug(
                    "[NO IMAGES FOUND] Page %d has no embedded image assets — OCR will not be initialized.",
                    page_num,
                )
            else:
                logger.info(
                    "[IMAGE DETECTED] Page %d contains %d embedded image(s) — routing to image extractors.",
                    page_num,
                    len(page_images),
                )

            for img_idx, img in enumerate(page_images):
                mime = "image/png"
                if img.name.lower().endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                elif img.name.lower().endswith(".webp"):
                    mime = "image/webp"

                logger.info("PDF image extraction | page=%d | image_index=%d", page_num, img_idx)

                # Run classifier
                image_type = "UNKNOWN"
                confidence = 0.0
                reason = "Default fallback"
                try:
                    import time
                    start_cls = time.time()
                    classifier_result = classify_image(img.data, mime)
                    record_performance_timing("classification_time", time.time() - start_cls)
                    
                    image_type = classifier_result.get("image_type", "UNKNOWN")
                    confidence = classifier_result.get("confidence", 0.0)
                    reason = classifier_result.get("reason", "Classification successful")
                except Exception as class_err:
                    logger.exception("Image classification failed on page %d, idx %d: %s", page_num, img_idx, class_err)

                logger.info("IMAGE PIPELINE classifier selected | type=%s", image_type)

                # Route to specialized extractors — all imports are deferred here
                # so PaddleOCR is only loaded when an image actually needs it.
                selected_extractor = ""
                extraction_result = None
                try:
                    if image_type == "TEXT_IMAGE":
                        from RAG.services.text_knowledge_extractor import extract_text_knowledge
                        selected_extractor = "extract_text_knowledge"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_text_knowledge(img.data, mime)
                    elif image_type == "CHART":
                        from RAG.services.chart_knowledge_extractor import extract_chart_knowledge
                        selected_extractor = "extract_chart_knowledge"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_chart_knowledge(img.data, mime)
                    elif image_type == "DIAGRAM":
                        from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge
                        selected_extractor = "extract_diagram_knowledge"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = extract_diagram_knowledge(img.data, mime)
                    elif image_type == "MIXED":
                        from RAG.routes.validation import run_visual_understanding_logic_on_bytes
                        selected_extractor = "run_visual_understanding_logic"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        extraction_result = run_visual_understanding_logic_on_bytes(img.data, mime)
                    elif image_type == "NATURAL_IMAGE":
                        selected_extractor = "describe_image_with_gemini"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        desc = describe_image_with_gemini(img.data, mime)
                        extraction_result = {
                            "summary": desc,
                            "rich_text_representation": f"# Natural Image Description\n\n## Gemini Vision Summary:\n{desc}\n"
                        }
                    else:
                        selected_extractor = "describe_image_with_gemini"
                        logger.info("IMAGE PIPELINE extractor selected | extractor=%s", selected_extractor)
                        desc = describe_image_with_gemini(img.data, mime)
                        extraction_result = {
                            "summary": desc,
                            "rich_text_representation": f"# Image Description\n\n## Gemini Vision Summary:\n{desc}\n"
                        }
                    logger.info("IMAGE PIPELINE extraction completed | extractor=%s", selected_extractor)
                except Exception as ext_err:
                    logger.exception("Image extractor %s failed on page %d, idx %d: %s", selected_extractor, page_num, img_idx, ext_err)
                    # Fallback to describe image
                    try:
                        selected_extractor = "describe_image_with_gemini (fallback)"
                        desc = describe_image_with_gemini(img.data, mime)
                        extraction_result = {
                            "summary": desc,
                            "rich_text_representation": f"# Image Description (Fallback after error)\n\n## Gemini Vision Summary:\n{desc}\n"
                        }
                    except Exception as fallback_err:
                        logger.exception("Fallback image describer also failed: %s", fallback_err)
                        extraction_result = {
                            "summary": "Failed to process visual content.",
                            "rich_text_representation": "# Visual Element Description Failed\n"
                        }

                # Construct normalized chunk
                rich_text = extraction_result.pop("rich_text_representation", "")
                normalized = normalize_knowledge_chunk(
                    document_name=filename,
                    page_number=page_num,
                    asset_type="image",
                    classification_type=image_type,
                    extractor_used=selected_extractor,
                    knowledge_object=extraction_result,
                    rich_text_representation=rich_text
                )
                chunks.append(normalized)
        except Exception as img_err:
            logger.exception("Failed to extract images on page %d: %s", page_num, img_err)
        finally:
            execution_stage_var.reset(stage_token_img)

        # ----------------------------------------------------
        # 3. Standard Text Chunker
        # ----------------------------------------------------
        stage_token_txt = execution_stage_var.set("text_extraction")
        try:
            page_text = plumber_page.extract_text() or ""
            if page_text.strip():
                logger.info("PDF text extraction | page=%d", page_num)
                text_chunks = split_text(page_text)
                for chunk_text in text_chunks:
                    normalized = normalize_knowledge_chunk(
                        document_name=filename,
                        page_number=page_num,
                        asset_type="text",
                        classification_type="text",
                        extractor_used="split_text",
                        knowledge_object={"text": chunk_text},
                        rich_text_representation=chunk_text
                    )
                    chunks.append(normalized)
        except Exception as text_err:
            logger.exception("Failed to extract text chunks on page %d: %s", page_num, text_err)
        finally:
            execution_stage_var.reset(stage_token_txt)

        logger.info("PDF page processing complete | page=%d | file=%s", page_num, filename)

    try:
        pdf.close()
    except Exception:
        pass

    logger.info("PDF orchestration and normalization complete | file=%s | extracted_chunks=%d", filename, len(chunks))
    return chunks

