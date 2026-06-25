import os
import logging
import base64
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from RAG.ingestion import describe_image_with_gemini

# Setup logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PaddleOCR Singleton — lazy-initialized, reused across all requests
# ---------------------------------------------------------------------------
_paddle_ocr_instance = None


def _get_paddle_ocr():
    """
    Returns the shared PaddleOCR engine instance, initializing it on first use.

    This singleton ensures PaddleOCR models are:
      - Never loaded at import time (no startup overhead)
      - Loaded exactly once when the first OCR request arrives
      - Reused for all subsequent requests (no per-call re-initialization)
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        logger.debug("[OCR INITIALIZED] PaddleOCR singleton does not exist — loading models now.")
        from paddleocr import PaddleOCR
        try:
            _paddle_ocr_instance = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False, enable_mkldnn=False
            )
        except Exception:
            try:
                _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)
            except Exception:
                _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
        logger.info("[OCR INITIALIZED] PaddleOCR singleton created and cached successfully.")
    else:
        logger.debug("[OCR INITIALIZED] Reusing existing PaddleOCR singleton — skipping model load.")
    return _paddle_ocr_instance

class MixedImageKnowledge(BaseModel):
    sections: List[str] = Field(description="List of logical sections or layout blocks identified in the infographic.")
    headings: List[str] = Field(description="List of main headings and subheadings extracted from the image.")
    labels: List[str] = Field(description="List of text labels, annotations, or callouts found in the infographic.")
    process_flow: List[str] = Field(description="Sequence of steps representing any process or chronological flow depicted.")
    key_takeaways: List[str] = Field(description="List of core messages, facts, or takeaways from the infographic.")
    summary: str = Field(description="A unified explanation combining text and visual layout elements.")

class NaturalImageKnowledge(BaseModel):
    scene_type: str = Field(description="The type of scene, e.g., 'outdoor', 'indoor', 'studio', 'aerial', 'close-up'.")
    objects: List[str] = Field(description="List of key physical objects, products, or elements detected in the photograph.")
    environment: str = Field(description="The context or environment description, e.g., 'urban street', 'electronics lab', 'nature park'.")
    activities: List[str] = Field(description="List of activities, actions, or states occurring in the image.")
    summary: str = Field(description="A concise summary description of the overall photograph.")


def run_ocr_with_diagnostics(image_path: str) -> dict:
    """Performs OCR audit on the image path, attempting PaddleOCR first (via singleton), then EasyOCR.
    Returns audit metrics dictionary or raises RuntimeError if both fail."""
    import time
    paddle_error = None
    easy_error = None

    # 1. Attempt PaddleOCR via singleton (no re-initialization overhead)
    logger.info("OCR Audit: Requesting PaddleOCR singleton...")
    try:
        ocr = _get_paddle_ocr()
        logger.info("OCR Audit: PaddleOCR singleton acquired. Running inference...") 
        
        t0 = time.perf_counter()
        try:
            result = ocr.ocr(image_path, cls=True)
        except Exception as te:
            if "cls" in str(te) or "unexpected keyword argument" in str(te):
                result = ocr.ocr(image_path)
            else:
                raise
        ocr_elapsed = time.perf_counter() - t0
        
        # Log the raw output structure
        logger.info(f"OCR Audit: PaddleOCR raw output structure: {result}")
        
        text_lines = []
        confidences = []
        blocks_detected = 0
        
        if result and result[0]:
            first_item = result[0]
            is_dict_like = isinstance(first_item, dict) or (hasattr(first_item, "keys") and hasattr(first_item, "get"))
            
            if is_dict_like and "rec_texts" in first_item.keys():
                rec_texts = first_item.get("rec_texts", [])
                rec_scores = first_item.get("rec_scores", [])
                for text, score in zip(rec_texts, rec_scores):
                    text_lines.append(text)
                    confidences.append(score)
                    blocks_detected += 1
            elif isinstance(first_item, list):
                for line in first_item:
                    if isinstance(line, list) and len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 1:
                        text_lines.append(line[1][0])
                        confidences.append(line[1][1])
                        blocks_detected += 1
                
        raw_text = "\n".join(text_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"OCR Audit: PaddleOCR success. Engine: PaddleOCR, Blocks: {blocks_detected}, Avg Confidence: {avg_conf:.4f}, Text Length: {len(raw_text)}, Execution Time: {ocr_elapsed:.4f}s")
        logger.info(f"OCR Audit: First 300 characters of extracted text:\n{raw_text[:300]}")
        logger.info(
            "OCR complete | engine=%s | execution_time=%.4f | confidence=%.4f | blocks=%d | first_300=%r",
            "PaddleOCR", ocr_elapsed, avg_conf, blocks_detected, raw_text[:300]
        )
        
        return {
            "ocr_engine_used": "PaddleOCR",
            "ocr_available": True,
            "ocr_raw_text": raw_text,
            "ocr_text_length": len(raw_text),
            "ocr_blocks_detected": blocks_detected,
            "average_confidence": round(avg_conf, 4),
            "ocr_execution_time": round(ocr_elapsed, 4),
        }
    except Exception as e:
        paddle_error = str(e)
        logger.warning(f"OCR Audit: PaddleOCR failed. Error: {paddle_error}")
        
    # 2. Attempt EasyOCR
    logger.info("OCR Audit: Attempting to load EasyOCR package...")
    try:
        import easyocr
        logger.info("OCR Audit: EasyOCR package loaded successfully. Initializing reader...")
        reader = easyocr.Reader(['en'])
        logger.info("OCR Audit: Running EasyOCR inference...")
        
        t0 = time.perf_counter()
        result = reader.readtext(image_path)
        ocr_elapsed = time.perf_counter() - t0
        
        # Log raw output structure
        logger.info(f"OCR Audit: EasyOCR raw output structure: {result}")
        
        text_lines = []
        confidences = []
        blocks_detected = 0
        
        if result:
            for line in result:
                text_lines.append(line[1])
                confidences.append(line[2])
                blocks_detected += 1
                
        raw_text = "\n".join(text_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"OCR Audit: EasyOCR success. Engine: EasyOCR, Blocks: {blocks_detected}, Avg Confidence: {avg_conf:.4f}, Text Length: {len(raw_text)}, Execution Time: {ocr_elapsed:.4f}s")
        logger.info(f"OCR Audit: First 300 characters of extracted text:\n{raw_text[:300]}")
        logger.info(
            "OCR complete | engine=%s | execution_time=%.4f | confidence=%.4f | blocks=%d | first_300=%r",
            "EasyOCR", ocr_elapsed, avg_conf, blocks_detected, raw_text[:300]
        )
        
        return {
            "ocr_engine_used": "EasyOCR",
            "ocr_available": True,
            "ocr_raw_text": raw_text,
            "ocr_text_length": len(raw_text),
            "ocr_blocks_detected": blocks_detected,
            "average_confidence": round(avg_conf, 4),
            "ocr_execution_time": round(ocr_elapsed, 4),
        }
    except Exception as e:
        easy_error = str(e)
        logger.warning(f"OCR Audit: EasyOCR failed. Error: {easy_error}")
        
    # 3. Both failed
    err_msg = f"OCR processing failed. PaddleOCR: {paddle_error}. EasyOCR: {easy_error}."
    logger.error(f"OCR Audit: {err_msg}")
    raise RuntimeError(err_msg)

def run_ocr_logic(image_path: str) -> str:
    """Performs OCR on the image path using PaddleOCR with EasyOCR fallback."""
    try:
        diagnostics = run_ocr_with_diagnostics(image_path)
        return diagnostics["ocr_raw_text"]
    except Exception as e:
        logger.error(f"run_ocr_logic failed: {e}")
        raise

def run_visual_understanding_logic(image_path: str, ext: str) -> dict:
    """Runs OCR + Gemini Vision structured output to generate a structured infographic explanation."""
    # 1. OCR text
    try:
        ocr_text = run_ocr_logic(image_path)
    except Exception as ocr_err:
        logger.warning(f"OCR failed in visual-understanding: {ocr_err}")
        ocr_text = "[OCR extraction failed or not available]"
        
    # 2. Vision execution using Gemini structured output
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            
        mime_type = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
            
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(MixedImageKnowledge)
        
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Analyze the provided infographic/mixed image. Below is the OCR text extracted from the image to assist you:\n"
            f"{ocr_text}\n\n"
            "Examine the image structure and extract sections, headings, text labels, any process flow, key takeaways, and a comprehensive summary."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        
        res = structured_chat.invoke([message])
        
        takeaways_str = "\n".join([f"- {t}" for t in res.key_takeaways])
        sections_str = "\n".join([f"- {s}" for s in res.sections])
        headings_str = "\n".join([f"- {h}" for h in res.headings])
        labels_str = "\n".join([f"- {l}" for l in res.labels])
        flow_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(res.process_flow)])
        
        rich_text = (
            f"# Mixed Image / Infographic Analysis\n\n"
            f"## Key Takeaways:\n"
            f"{takeaways_str if takeaways_str else '[No takeaways identified]'}\n\n"
            f"## Sections:\n"
            f"{sections_str if sections_str else '[No sections identified]'}\n\n"
            f"## Headings:\n"
            f"{headings_str if headings_str else '[No headings identified]'}\n\n"
            f"## Process Flow:\n"
            f"{flow_str if flow_str else '[No process flow identified]'}\n\n"
            f"## Labels:\n"
            f"{labels_str if labels_str else '[No labels identified]'}\n\n"
            f"## Summary:\n"
            f"{res.summary}\n"
        )
        
        return {
            "image_type": "MIXED",
            "sections": res.sections,
            "headings": res.headings,
            "labels": res.labels,
            "process_flow": res.process_flow,
            "key_takeaways": res.key_takeaways,
            "summary": res.summary,
            # Legacy keys for backwards compatibility:
            "ocr_text": ocr_text,
            "combined_understanding": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error(f"Error in run_visual_understanding_logic: {e}")
        raise RuntimeError(f"Visual understanding logic failed: {str(e)}")

def extract_natural_image_knowledge(image_bytes: bytes, mime_type: str) -> dict:
    """Uses Gemini Vision structured output to extract detailed visual understanding from a natural photo."""
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(NaturalImageKnowledge)
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Analyze the provided photograph or real-world image. Extract scene characteristics: "
            "scene type, key physical objects, environment context, activities occurring in the image, and a summary."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        
        res = structured_chat.invoke([message])
        
        objects_str = "\n".join([f"- {obj}" for obj in res.objects])
        activities_str = "\n".join([f"- {act}" for act in res.activities])
        
        rich_text = (
            f"# Natural Image Description\n\n"
            f"- **Scene Type**: {res.scene_type}\n"
            f"- **Environment**: {res.environment}\n\n"
            f"## Summary:\n"
            f"{res.summary}\n\n"
            f"## Key Objects:\n"
            f"{objects_str if objects_str else '[No objects detected]'}\n\n"
            f"## Activities:\n"
            f"{activities_str if activities_str else '[No activities detected]'}\n"
        )
        
        return {
            "image_type": "NATURAL_IMAGE",
            "scene_type": res.scene_type,
            "objects": res.objects,
            "environment": res.environment,
            "activities": res.activities,
            "summary": res.summary,
            # Legacy / back-compat keys:
            "description": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error(f"Error in extract_natural_image_knowledge: {e}")
        raise RuntimeError(f"Natural image extraction failed: {str(e)}")

