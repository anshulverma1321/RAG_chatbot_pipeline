import base64
import logging
from enum import Enum
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image Type Labels
# ---------------------------------------------------------------------------

class ImageType(str, Enum):
    """Canonical label set for classifying document and visual images."""
    TEXT_IMAGE    = "TEXT_IMAGE"     # Screenshots, scanned pages, text-heavy images
    CHART         = "CHART"          # Bar, line, pie, histogram, area charts
    DIAGRAM       = "DIAGRAM"        # Flowcharts, architecture diagrams, UML, topology
    NATURAL_IMAGE = "NATURAL_IMAGE"  # Photographs, product shots, scenes, objects
    MIXED         = "MIXED"          # Infographics, chart+text combos, multi-category
    UNKNOWN       = "UNKNOWN"        # Low-confidence or unrecognisable content


# ---------------------------------------------------------------------------
# Pydantic Schema for Structured Gemini Output
# ---------------------------------------------------------------------------

class ImageClassification(BaseModel):
    """Structured classification result returned by the Gemini VLM."""
    image_type: ImageType = Field(
        description=(
            "The primary category of the image. Must be one of: "
            "TEXT_IMAGE, CHART, DIAGRAM, NATURAL_IMAGE, MIXED, UNKNOWN. "
            "Choose TEXT_IMAGE for screenshots, scanned text pages, or images where text is the dominant element. "
            "Choose CHART for any statistical graph (bar, line, pie, histogram, area). "
            "Choose DIAGRAM for flowcharts, architecture diagrams, UML, network or system topologies. "
            "Choose NATURAL_IMAGE for photographs, product images, objects, or scenes with no significant text or data overlays. "
            "Choose MIXED when the image clearly combines two or more of the above categories (e.g., a chart embedded in a screenshot, or an infographic). "
            "Choose UNKNOWN only when confidence is very low and no other category fits."
        )
    )
    confidence: float = Field(
        description=(
            "A confidence score between 0.0 and 1.0 representing how certain the classification is. "
            "1.0 means completely certain; below 0.5 should lean toward UNKNOWN."
        ),
        ge=0.0,
        le=1.0
    )
    reason: str = Field(
        description=(
            "A concise, one-to-three sentence explanation of why this classification was chosen. "
            "Reference specific visual evidence observed in the image (e.g., 'The image contains a vertical bar chart "
            "with labeled axes and a legend indicating quarterly revenue.') "
            "Do not include generic or conversational filler text."
        )
    )


# ---------------------------------------------------------------------------
# Public Classifier Function
# ---------------------------------------------------------------------------

def classify_image(image_bytes: bytes, mime_type: str) -> dict:
    """
    Classifies a given image using Gemini 3.1 Flash Lite structured output.

    Determines the primary visual category of the image from the following
    label set: TEXT_IMAGE, CHART, DIAGRAM, NATURAL_IMAGE, MIXED, UNKNOWN.

    Args:
        image_bytes: Raw bytes of the image to classify.
        mime_type:   MIME type string, e.g. 'image/png', 'image/jpeg', 'image/webp'.

    Returns:
        A dict with keys:
            - "image_type"  (str): One of the ImageType enum values.
            - "confidence"  (float): Classification confidence score between 0.0 and 1.0.
            - "reason"      (str): Human-readable explanation of the classification decision.

    Raises:
        RuntimeError: If Gemini classification fails and no fallback is possible.
    """
    logger.info(
        "classify_image called | mime_type=%s | image_size=%d bytes",
        mime_type,
        len(image_bytes)
    )

    try:
        # ----------------------------------------------------------------
        # 1. Initialise model with structured output schema
        # ----------------------------------------------------------------
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(ImageClassification)

        logger.debug("Gemini structured output schema bound to ImageClassification.")

        # ----------------------------------------------------------------
        # 2. Encode image to base64 data URL
        # ----------------------------------------------------------------
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"

        logger.debug("Image encoded to base64 data URL successfully.")

        # ----------------------------------------------------------------
        # 3. Compose the classification prompt
        # ----------------------------------------------------------------
        prompt = (
            "You are an expert visual document analyst. Your task is to classify the provided image "
            "into exactly one of the following categories:\n\n"
            "  • TEXT_IMAGE  – The image is dominated by text content. Examples: a screenshot of a "
            "document, a scanned page of a book, a slide with bullet points, a code snippet screenshot.\n\n"
            "  • CHART       – The image is a statistical or data visualisation. Examples: bar chart, "
            "line graph, pie chart, histogram, area chart, scatter plot, heat map.\n\n"
            "  • DIAGRAM     – The image depicts a structural or logical model. Examples: flowchart, "
            "software architecture diagram, UML class diagram, network topology, ER diagram.\n\n"
            "  • NATURAL_IMAGE – The image is a real-world photograph or product/scene illustration "
            "with no significant overlaid data or text. Examples: a photo of a building, a product "
            "catalogue image, a portrait, a landscape scene.\n\n"
            "  • MIXED       – The image contains a clear combination of two or more of the above "
            "categories. Examples: an infographic mixing charts and text, a screenshot containing an "
            "embedded diagram, a slide with both a graph and a paragraph.\n\n"
            "  • UNKNOWN     – The content is ambiguous, corrupted, or completely unrecognisable, "
            "and no other category applies with reasonable confidence (confidence < 0.5).\n\n"
            "Instructions:\n"
            "1. Examine the image carefully before deciding.\n"
            "2. Select the single best-fitting category.\n"
            "3. Assign a confidence score between 0.0 (no certainty) and 1.0 (absolute certainty).\n"
            "4. Provide a concise factual reason citing specific visual evidence from the image.\n"
            "5. Do not guess or hallucinate. If genuinely unsure, use UNKNOWN with a low confidence score."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )

        # ----------------------------------------------------------------
        # 4. Invoke Gemini and capture the structured response
        # ----------------------------------------------------------------
        logger.info("Sending image classification request to Gemini VLM...")
        result: ImageClassification = structured_chat.invoke([message])

        logger.info(
            "Gemini classification received | image_type=%s | confidence=%.2f",
            result.image_type.value,
            result.confidence
        )
        logger.debug("Classification reason: %s", result.reason)

        # ----------------------------------------------------------------
        # 5. Return normalised dict payload
        # ----------------------------------------------------------------
        return {
            "image_type": result.image_type.value,
            "confidence": round(result.confidence, 4),
            "reason": result.reason.strip()
        }

    except Exception as e:
        logger.error(
            "classify_image failed | mime_type=%s | error_type=%s | error=%s",
            mime_type,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise RuntimeError(f"Image classification failed: {str(e)}") from e


# ---------------------------------------------------------------------------
# Standalone Test Function
# ---------------------------------------------------------------------------

def _run_standalone_test():
    """
    Standalone test that can be executed directly (python image_classifier.py).

    Reads a test image file from disk (or downloads a small placeholder),
    runs classify_image(), and prints the result to stdout.

    Usage:
        python RAG/services/image_classifier.py
        python RAG/services/image_classifier.py --path /path/to/image.png
    """
    import os
    import sys
    import json
    from dotenv import load_dotenv

    # Resolve project root and load .env so GEMINI_API_KEY is available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # Configure basic logging so log lines are visible during the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY is not set. Add it to your .env file or environment.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve the test image path from CLI arg or fall back to test_tts.wav
    # sibling in project root (we only need any image file to exercise the call)
    # ------------------------------------------------------------------
    if len(sys.argv) > 2 and sys.argv[1] == "--path":
        image_path = sys.argv[2]
    else:
        # Default: look for any .png or .jpg in the project root for a quick test
        candidates = []
        for fname in os.listdir(project_root):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                candidates.append(os.path.join(project_root, fname))

        if not candidates:
            print(
                "[WARN] No image file found in project root.\n"
                "       Run with:  python RAG/services/image_classifier.py --path /your/image.png"
            )
            sys.exit(0)

        image_path = candidates[0]

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    print(f"\n{'='*60}")
    print(f"  Image Classifier — Standalone Test")
    print(f"{'='*60}")
    print(f"  File      : {image_path}")
    print(f"  MIME Type : {mime_type}")
    print(f"  File Size : {os.path.getsize(image_path):,} bytes")
    print(f"{'='*60}\n")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    try:
        result = classify_image(image_bytes, mime_type)
        print("Classification Result:")
        print(json.dumps(result, indent=2))
    except RuntimeError as e:
        print(f"[ERROR] Classification failed: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Test completed successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _run_standalone_test()
