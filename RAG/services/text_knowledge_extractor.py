import os
import tempfile
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
# NOTE: run_ocr_with_diagnostics is imported inside run_ocr_on_bytes_with_diagnostics() to avoid
# eager PaddleOCR initialization at import time (lazy-loading optimization).

logger = logging.getLogger(__name__)

class TextMetaExtraction(BaseModel):
    document_type: str = Field(description="The inferred type of document (e.g. 'article', 'invoice', 'report', 'code_snippet', etc.).")
    key_points: List[str] = Field(description="List of 3 to 5 key points or main ideas extracted from the text.")
    entities: List[str] = Field(description="List of key entities mentioned (people, organizations, places, products, dates, APIs).")

def extract_text_metadata(text: str) -> TextMetaExtraction:
    """Uses Gemini to extract document type, key points, and entities from clean text."""
    if not text.strip():
        return TextMetaExtraction(document_type="unspecified", key_points=[], entities=[])
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(TextMetaExtraction)
        prompt = (
            "Analyze the following text extracted from a document image. Identify the document type, "
            "extract 3 to 5 main key points, and list key entities (organizations, locations, technical terms, names, dates, APIs).\n\n"
            f"Text:\n{text}"
        )
        return structured_chat.invoke(prompt)
    except Exception as e:
        logger.error(f"Failed to extract text metadata: {e}")
        return TextMetaExtraction(document_type="unspecified", key_points=[], entities=[])


def run_ocr_on_bytes_with_diagnostics(image_bytes: bytes, mime_type: str) -> dict:
    """Helper to run OCR on raw image bytes via a temporary file and gather diagnostics.
    
    NOTE: run_ocr_with_diagnostics (and therefore image_intelligence / PaddleOCR) is imported
    here at call time, not at module load time, so the OCR stack is never
    initialized unless a TEXT_IMAGE is actually being processed.
    """
    from RAG.services.image_intelligence import run_ocr_with_diagnostics  # deferred — lazy OCR load
    ext = ".png"
    if mime_type == "image/jpeg":
        ext = ".jpg"
    elif mime_type == "image/webp":
        ext = ".webp"
        
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(temp_fd, 'wb') as tmp:
            tmp.write(image_bytes)
        diagnostics = run_ocr_with_diagnostics(temp_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return diagnostics

def clean_ocr_text_with_gemini_diagnostics(raw_text: str) -> tuple[str, dict]:
    """Uses Gemini to clean obvious OCR spelling and spacing issues, returning cleaned text and diagnostics.
    Does not swallow exceptions, but logs them with full context."""
    if not raw_text.strip():
        return "", {
            "gemini_cleaning_execution_time": 0.0,
            "gemini_response_type": "NoneType"
        }
        
    import time
    t0 = time.perf_counter()
    content_type = "NoneType"
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        prompt = (
            "You are an assistant that cleans up raw OCR text. The input text is from a document scanned by an OCR engine.\n"
            "Your task is to:\n"
            "1. Correct obvious spelling mistakes and character recognition errors (e.g. '1nstnce' -> 'instance').\n"
            "2. Fix incorrect hyphenation at line breaks and restore proper word spacing.\n"
            "3. Maintain the exact original semantic content, definitions, numbers, names, and terminology.\n"
            "4. Do NOT summarize, rewrite, or add any meta-commentary.\n\n"
            f"Raw OCR Text:\n{raw_text}\n\n"
            "Cleaned Text:"
        )
        response = chat.invoke(prompt)
        content = response.content
        content_type = type(content).__name__
        
        logger.info(
            "Gemini response received | type=%s | raw_len=%d",
            content_type,
            len(raw_text)
        )
        
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
            
        cleaned_text = content.strip() if content else raw_text
        elapsed = time.perf_counter() - t0
        
        logger.info(
            "Gemini cleaning complete | execution_time=%.4f | response_type=%s",
            elapsed,
            content_type
        )
        
        return cleaned_text, {
            "gemini_cleaning_execution_time": round(elapsed, 4),
            "gemini_response_type": content_type
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("Gemini OCR cleaning failed: %s", e, exc_info=True)
        return raw_text, {
            "gemini_cleaning_execution_time": round(elapsed, 4),
            "gemini_response_type": content_type,
            "error": str(e)
        }

def clean_ocr_text_with_gemini(raw_text: str) -> str:
    """Uses Gemini to clean obvious OCR spelling and spacing issues while preserving literal text."""
    cleaned_text, _ = clean_ocr_text_with_gemini_diagnostics(raw_text)
    return cleaned_text

def extract_text_knowledge(image_bytes: bytes, mime_type: str) -> dict:
    """Extracts raw text, cleans it using Gemini, and compiles detailed diagnostics."""
    logger.info("[IMAGE DETECTED] extract_text_knowledge called | mime_type=%s | size=%d bytes", mime_type, len(image_bytes))
    # 1. Run OCR with Diagnostics
    diagnostics = run_ocr_on_bytes_with_diagnostics(image_bytes, mime_type)
    raw_ocr = diagnostics["ocr_raw_text"]
    
    # 2. Clean Text
    cleaned_text, clean_diag = clean_ocr_text_with_gemini_diagnostics(raw_ocr)
    
    # 3. Word Count
    word_count = len(cleaned_text.split()) if cleaned_text else 0
    
    # 4. Extract Structured Metadata (document type, key points, entities)
    meta = extract_text_metadata(cleaned_text)
    
    # 5. Rich Text Representation
    rich_text = (
        f"# Document Text Knowledge Extraction\n\n"
        f"- **Document Type**: {meta.document_type}\n"
        f"- **Word Count**: {word_count}\n\n"
        f"## Content:\n"
        f"{cleaned_text if cleaned_text else '[No text extracted]'}\n\n"
    )
    if meta.key_points:
        rich_text += "## Key Points:\n" + "\n".join([f"- {kp}" for kp in meta.key_points]) + "\n\n"
    if meta.entities:
        rich_text += "## Entities:\n" + "\n".join([f"- {ent}" for ent in meta.entities]) + "\n\n"
    
    rich_text += (
        f"---\n"
        f"Metadata:\n"
        f"- Image Type: text_image\n"
        f"- Word Count: {word_count}\n"
    )
    
    return {
        "image_type": "text_image",
        "document_type": meta.document_type,
        "extracted_text": cleaned_text,
        "cleaned_text": cleaned_text,
        "key_points": meta.key_points,
        "entities": meta.entities,
        "word_count": word_count,
        "rich_text_representation": rich_text,
        "ocr_engine_used": diagnostics["ocr_engine_used"],
        "ocr_available": diagnostics["ocr_available"],
        "ocr_raw_text": raw_ocr,
        "ocr_text_length": diagnostics["ocr_text_length"],
        "ocr_blocks_detected": diagnostics["ocr_blocks_detected"],
        "average_confidence": diagnostics["average_confidence"],
        "ocr_execution_time": diagnostics.get("ocr_execution_time", 0.0),
        "gemini_cleaning_execution_time": clean_diag.get("gemini_cleaning_execution_time", 0.0),
        "gemini_response_type": clean_diag.get("gemini_response_type", "NoneType")
    }

