import os
import base64
import logging
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
# NOTE: run_ocr_logic is imported inside run_ocr_on_bytes() to avoid eager
# PaddleOCR initialization at import time (lazy-loading optimization).

logger = logging.getLogger(__name__)

class DiagramEdge(BaseModel):
    from_node: str = Field(description="The label or name of the source node/component.", alias="from")
    to_node: str = Field(description="The label or name of the destination node/component.", alias="to")
    label_or_relationship: Optional[str] = Field(None, description="The transition label/condition (like 'Yes', 'No') or relationship description (like 'HTTPS request').")

    class Config:
        populate_by_name = True

class DiagramKnowledge(BaseModel):
    diagram_type: str = Field(description="The category of diagram, e.g. 'flowchart', 'architecture_diagram', or 'other'.")
    nodes: List[str] = Field(description="List of all nodes, modules, services, user roles, databases, or process steps in the diagram.")
    relationships: List[DiagramEdge] = Field(description="List of all directed edges, dependencies, interactions, or transitions between components.")
    workflow: List[str] = Field(description="Sequence of steps representing the operational process flow or data flow path.")
    components: List[str] = Field(description="List of physical or logical components/modules shown in the diagram.")
    summary: str = Field(description="A step-by-step explanation of the flowchart logic or system architecture data flow.")


def run_ocr_on_bytes(image_bytes: bytes, mime_type: str) -> str:
    """Helper to run PaddleOCR on raw image bytes via a temporary file.
    
    NOTE: run_ocr_logic (and therefore image_intelligence / PaddleOCR) is imported
    here at call time, not at module load time, so the OCR stack is never
    initialized unless a diagram image is actually being processed.
    """
    from RAG.services.image_intelligence import run_ocr_logic  # deferred — lazy OCR load
    ext = ".png"
    if mime_type == "image/jpeg":
        ext = ".jpg"
    elif mime_type == "image/webp":
        ext = ".webp"
        
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(temp_fd, 'wb') as tmp:
            tmp.write(image_bytes)
        ocr_text = run_ocr_logic(temp_path)
    except Exception as e:
        logger.error(f"OCR execution failed on diagram temp file: {e}")
        ocr_text = ""
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return ocr_text

def extract_diagram_knowledge(image_bytes: bytes, mime_type: str) -> dict:
    """Uses OCR and Gemini Vision to map flowcharts and system architectures into nodes, edges, and descriptions."""
    logger.info("[IMAGE DETECTED] extract_diagram_knowledge called | mime_type=%s | size=%d bytes", mime_type, len(image_bytes))
    try:
        # 1. OCR preprocessing step
        ocr_text = run_ocr_on_bytes(image_bytes, mime_type)
        
        # 2. Vision execution using Gemini
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(DiagramKnowledge)
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Analyze this diagram image (which could be a flowchart or a system architecture diagram) "
            "and extract its topology and structure.\n"
            "Your output must contain:\n"
            "1. The type of diagram (either 'flowchart' or 'architecture_diagram').\n"
            "2. All nodes (representing steps, components, modules, actors, or databases).\n"
            "3. All directed connections (relationships) representing data flow, transitions, or relationship directions.\n"
            "4. A sequence of steps (workflow) representing the operational process flow.\n"
            "5. A list of physical/logical components or subsystems.\n"
            "6. A complete textual description / summary of the process flowchart logic or architecture data flow."
        )
        if ocr_text:
            prompt += f"\n\nHere is text extracted from the image by OCR to assist you:\n{ocr_text}"
            
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        
        res = structured_chat.invoke([message])
        
        # Format the rich text representation for vector indexing
        nodes_str = "\n".join([f"- {node}" for node in res.nodes])
        components_str = "\n".join([f"- {comp}" for comp in res.components])
        workflow_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(res.workflow)])
        
        edges_list = []
        for edge in res.relationships:
            rel = f" --[{edge.label_or_relationship}]--> " if edge.label_or_relationship else " ----> "
            edges_list.append(f"- **{edge.from_node}**{rel}**{edge.to_node}**")
        edges_str = "\n".join(edges_list)
        
        rich_text = (
            f"# Diagram Structure & Flow Extraction\n\n"
            f"- **Diagram Type**: {res.diagram_type}\n\n"
            f"## Connections / Interactions:\n"
            f"{edges_str if edges_str else '[No connections mapped]'}\n\n"
            f"## Components:\n"
            f"{components_str if components_str else '[No components identified]'}\n\n"
            f"## Nodes:\n"
            f"{nodes_str if nodes_str else '[No nodes identified]'}\n\n"
            f"## Workflow:\n"
            f"{workflow_str if workflow_str else '[No workflow mapped]'}\n\n"
            f"## Process flow / Architectural description:\n"
            f"{res.summary}\n"
        )
        
        # Format relationship list with alias keys for API schema consistency
        formatted_relationships = [
            {"from": edge.from_node, "to": edge.to_node, "label_or_relationship": edge.label_or_relationship}
            for edge in res.relationships
        ]
        
        return {
            "image_type": "diagram",
            "diagram_type": res.diagram_type,
            "nodes": res.nodes,
            "relationships": formatted_relationships,
            "workflow": res.workflow,
            "components": res.components,
            "summary": res.summary,
            # Backward compatibility keys:
            "edges": formatted_relationships,
            "description": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error(f"Error in extract_diagram_knowledge: {e}")
        raise RuntimeError(f"Diagram knowledge extraction failed: {str(e)}")
