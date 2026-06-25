import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NormalizedKnowledgeChunk(BaseModel):
    """Universal normalized knowledge representation for all types of document assets."""
    chunk_id: str = Field(..., description="Unique UUID string identifying this chunk.")
    document_name: str = Field(..., description="Name of the source document.")
    page_number: int = Field(..., description="1-based page number.")
    asset_type: str = Field(..., description="Broad asset classification: 'text', 'image', or 'table'.")
    classification_type: str = Field(..., description="Sub-type classifier label (e.g. CHART, TABLE_FINANCIAL, or 'text').")
    extractor_used: str = Field(..., description="The name of the extraction pipeline function executed.")
    structured_knowledge: Dict[str, Any] = Field(..., description="The structured key-value payload returned by the extractor.")
    rich_text_representation: str = Field(..., description="Clean Markdown summary representation of the chunk.")
    embedding_text: str = Field(..., description="Intelligently formatted text optimized for semantic vectors.")
    metadata: Dict[str, Any] = Field(..., description="Unified metadata fields for vector index filtering.")


def _generate_embedding_text(
    asset_type: str,
    classification_type: str,
    knowledge_object: Dict[str, Any],
    rich_text: str
) -> str:
    """Intelligently generates embedding text matching specialized RAG patterns for high retrieval accuracy."""
    ctype = classification_type.upper()

    if asset_type == "text" or ctype == "TEXT":
        return knowledge_object.get("text", "") or rich_text or ""

    if ctype == "TEXT_IMAGE":
        return knowledge_object.get("cleaned_text") or knowledge_object.get("extracted_text") or knowledge_object.get("text") or rich_text or ""

    if ctype == "CHART":
        parts = []
        if knowledge_object.get("title"):
            parts.append(f"Chart Title: {knowledge_object['title']}")
        if knowledge_object.get("chart_type"):
            parts.append(f"Chart Type: {knowledge_object['chart_type']}")
        if knowledge_object.get("x_axis"):
            parts.append(f"X-Axis Labels/Variables: {knowledge_object['x_axis']}")
        if knowledge_object.get("y_axis"):
            parts.append(f"Y-Axis Labels/Values: {knowledge_object['y_axis']}")
        if knowledge_object.get("data_points"):
            dp_list = []
            for dp in knowledge_object["data_points"]:
                if isinstance(dp, dict):
                    dp_list.append(f"{dp.get('label', '')}: {dp.get('value', '')}")
            parts.append(f"Data Points: {', '.join(dp_list)}")
        if knowledge_object.get("trends"):
            parts.append(f"Key Trends: {'; '.join(knowledge_object['trends'])}")
        if knowledge_object.get("insights"):
            parts.append(f"Insights: {'; '.join(knowledge_object['insights'])}")
        if rich_text:
            parts.append(f"Summary:\n{rich_text}")
        return "\n".join(parts)

    if ctype == "DIAGRAM":
        parts = []
        if knowledge_object.get("diagram_type"):
            parts.append(f"Diagram Type: {knowledge_object['diagram_type']}")
        if knowledge_object.get("nodes"):
            parts.append(f"Nodes: {', '.join(knowledge_object['nodes'])}")
        if knowledge_object.get("relationships"):
            rel_list = []
            for r in knowledge_object["relationships"]:
                if isinstance(r, dict):
                    rel_list.append(f"{r.get('from', '')} -> {r.get('to', '')} ({r.get('label_or_relationship') or 'link'})")
            parts.append(f"Relationships: {', '.join(rel_list)}")
        if knowledge_object.get("workflow"):
            parts.append(f"Workflow Process Steps: {'; '.join(knowledge_object['workflow'])}")
        if knowledge_object.get("components"):
            parts.append(f"Components/Modules: {', '.join(knowledge_object['components'])}")
        if knowledge_object.get("summary"):
            parts.append(f"Process/Architecture Summary: {knowledge_object['summary']}")
        if rich_text:
            parts.append(f"Summary:\n{rich_text}")
        return "\n".join(parts)

    if ctype == "MIXED":
        parts = []
        if knowledge_object.get("sections"):
            parts.append(f"Infographic Sections: {', '.join(knowledge_object['sections'])}")
        if knowledge_object.get("headings"):
            parts.append(f"Headings: {', '.join(knowledge_object['headings'])}")
        if knowledge_object.get("labels"):
            parts.append(f"Text Labels: {', '.join(knowledge_object['labels'])}")
        if knowledge_object.get("process_flow"):
            parts.append(f"Process/Flow: {'; '.join(knowledge_object['process_flow'])}")
        if knowledge_object.get("key_takeaways"):
            parts.append(f"Key Takeaways: {'; '.join(knowledge_object['key_takeaways'])}")
        if knowledge_object.get("summary"):
            parts.append(f"Content Summary: {knowledge_object['summary']}")
        if rich_text:
            parts.append(f"Summary:\n{rich_text}")
        return "\n".join(parts)

    if asset_type == "table" or ctype.startswith("TABLE"):
        parts = []
        if knowledge_object.get("title"):
            parts.append(f"Table Title: {knowledge_object['title']}")
        if knowledge_object.get("summary"):
            parts.append(f"Table Summary: {knowledge_object['summary']}")
        if knowledge_object.get("headers"):
            parts.append(f"Headers: {', '.join(knowledge_object['headers'])}")

        # TABLE_FINANCIAL
        if knowledge_object.get("reporting_period"):
            parts.append(f"Reporting Period: {knowledge_object['reporting_period']}")
        if knowledge_object.get("currency"):
            parts.append(f"Currency: {knowledge_object['currency']}")
        if knowledge_object.get("financial_metrics"):
            metric_list = []
            for m in knowledge_object["financial_metrics"]:
                if isinstance(m, dict):
                    metric_list.append(f"{m.get('category', '')}: {m.get('values', '')}")
            parts.append(f"Financial Metrics: {', '.join(metric_list)}")
        if knowledge_object.get("key_financial_takeaways"):
            parts.append(f"Key Findings: {'; '.join(knowledge_object['key_financial_takeaways'])}")

        # TABLE_STATISTICAL
        if knowledge_object.get("variables"):
            parts.append(f"Variables Evaluated: {', '.join(knowledge_object['variables'])}")
        if knowledge_object.get("metrics"):
            parts.append(f"Metrics Reported: {', '.join(knowledge_object['metrics'])}")
        if knowledge_object.get("data_summary"):
            sum_list = []
            for s in knowledge_object["data_summary"]:
                if isinstance(s, dict):
                    sum_list.append(f"{s.get('variable_or_group', '')}: {s.get('metric_values', '')}")
            parts.append(f"Data Summary: {', '.join(sum_list)}")
        if knowledge_object.get("statistical_conclusions"):
            parts.append(f"Statistical Conclusions: {'; '.join(knowledge_object['statistical_conclusions'])}")

        # TABLE_TIMESERIES
        if knowledge_object.get("time_interval"):
            parts.append(f"Time Interval: {knowledge_object['time_interval']}")
        if knowledge_object.get("timestamps"):
            parts.append(f"Timestamps: {', '.join(knowledge_object['timestamps'])}")
        if knowledge_object.get("series_data"):
            series_list = []
            for s in knowledge_object["series_data"]:
                if isinstance(s, dict):
                    series_list.append(f"{s.get('metric_name', '')}: {s.get('values_over_time', '')}")
            parts.append(f"Series Data: {', '.join(series_list)}")
        if knowledge_object.get("trends_observed"):
            parts.append(f"Observed Trends: {'; '.join(knowledge_object['trends_observed'])}")

        # TABLE_COMPARISON
        if knowledge_object.get("entities_compared"):
            parts.append(f"Entities Compared: {', '.join(knowledge_object['entities_compared'])}")
        if knowledge_object.get("attributes_compared"):
            parts.append(f"Attributes Compared: {', '.join(knowledge_object['attributes_compared'])}")
        if knowledge_object.get("key_differences"):
            parts.append(f"Key Differences: {'; '.join(knowledge_object['key_differences'])}")

        if rich_text:
            parts.append(f"Summary:\n{rich_text}")
        return "\n".join(parts)

    if ctype == "NATURAL_IMAGE":
        parts = []
        if knowledge_object.get("scene_type"):
            parts.append(f"Scene Type: {knowledge_object['scene_type']}")
        if knowledge_object.get("environment"):
            parts.append(f"Environment Context: {knowledge_object['environment']}")
        if knowledge_object.get("objects"):
            parts.append(f"Detected Objects: {', '.join(knowledge_object['objects'])}")
        if knowledge_object.get("activities"):
            parts.append(f"Activities: {', '.join(knowledge_object['activities'])}")
        
        sum_desc = knowledge_object.get("summary") or knowledge_object.get("description") or ""
        if sum_desc:
            parts.append(f"Visual Description: {sum_desc}")
        if rich_text:
            parts.append(f"Summary:\n{rich_text}")
        return "\n".join(parts)

    # General Fallback
    fallback_desc = knowledge_object.get("summary") or knowledge_object.get("description") or knowledge_object.get("text") or ""
    parts = []
    if fallback_desc:
        parts.append(f"Description: {fallback_desc}")
    if rich_text:
        parts.append(f"Summary:\n{rich_text}")
    return "\n".join(parts) if parts else ""


def normalize_knowledge_chunk(
    document_name: str,
    page_number: int,
    asset_type: str,
    classification_type: str,
    extractor_used: str,
    knowledge_object: Dict[str, Any],
    rich_text_representation: Optional[str] = None
) -> NormalizedKnowledgeChunk:
    """
    Standardizes extraction metadata and structured details from any layout asset
    into a unified representation ready for downstream vector embeddings and storage.
    """
    import logging
    import time
    from RAG.logger import execution_stage_var, record_performance_timing
    
    stage_token = execution_stage_var.set("normalization")
    start_norm = time.time()
    try:
        norm_logger = logging.getLogger("RAG.ingestion")
        # 1. Generate unique chunk ID
        chunk_id = str(uuid.uuid4())

        # 2. Extract rich text representation if not provided
        kobj_copy = dict(knowledge_object)  # copy to avoid in-place mutation of the original
        rich_text = rich_text_representation
        if rich_text is None:
            rich_text = kobj_copy.pop("rich_text_representation", "")

        # 3. Create embedding optimized text
        embedding_text = _generate_embedding_text(
            asset_type=asset_type,
            classification_type=classification_type,
            knowledge_object=kobj_copy,
            rich_text=rich_text
        )

        norm_logger.info("NORMALIZATION chunk normalized | asset_type=%s", asset_type)
        norm_logger.info("NORMALIZATION embedding text generated | chunk_id=%s", chunk_id)

        # 4. Formulate unified metadata envelope
        metadata = {
            "document_name": document_name,
            "page_number": page_number,
            "asset_type": asset_type,
            "classification_type": classification_type,
            "extractor_used": extractor_used,
        }

        # Track normalization duration
        record_performance_timing("normalization_time", time.time() - start_norm)

        return NormalizedKnowledgeChunk(
            chunk_id=chunk_id,
            document_name=document_name,
            page_number=page_number,
            asset_type=asset_type,
            classification_type=classification_type,
            extractor_used=extractor_used,
            structured_knowledge=kobj_copy,
            rich_text_representation=rich_text,
            embedding_text=embedding_text,
            metadata=metadata
        )
    finally:
        execution_stage_var.reset(stage_token)
