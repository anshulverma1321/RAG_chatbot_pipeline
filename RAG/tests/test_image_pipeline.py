import os
import sys
import json
import time
import logging
import argparse

# ---------------------------------------------------------------------------
# Resolve project root and make it importable before any RAG imports
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# Logging setup — matches the style used across all existing test scripts
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 64

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))

def section(title: str):
    """Prints a formatted section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def elapsed_label(seconds: float) -> str:
    """Returns a human-readable elapsed time string."""
    return f"{seconds:.3f}s"


# ---------------------------------------------------------------------------
# MIME helper
# ---------------------------------------------------------------------------

def get_mime_type(image_path: str) -> str:
    """Derives the MIME type from the file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png":  "image/png",
    }
    return mime_map.get(ext, "image/png")


# ---------------------------------------------------------------------------
# Step 1 — Classification
# ---------------------------------------------------------------------------

def run_classification(image_bytes: bytes, mime_type: str) -> dict:
    """
    Runs classify_image() and returns the raw classification dict.
    Logs entry, exit, and elapsed time.
    """
    from RAG.services.image_classifier import classify_image

    section("STEP 1 — IMAGE CLASSIFICATION")
    logger.info("Starting classification | mime_type=%s | size=%d bytes", mime_type, len(image_bytes))

    t0 = time.perf_counter()
    result = classify_image(image_bytes, mime_type)
    elapsed = time.perf_counter() - t0

    logger.info(
        "Classification complete | image_type=%s | confidence=%.4f | elapsed=%s",
        result["image_type"],
        result["confidence"],
        elapsed_label(elapsed)
    )

    print(f"\n  Image Type   : {result['image_type']}")
    print(f"  Confidence   : {result['confidence']:.4f}")
    print(f"  Reason       : {result['reason']}")
    print(f"  Elapsed      : {elapsed_label(elapsed)}")

    result["_elapsed_classification"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Step 2 — Routed Extraction
# ---------------------------------------------------------------------------

def route_and_extract(
    image_type: str,
    image_bytes: bytes,
    mime_type: str,
    image_path: str
) -> dict:
    """
    Routes to the correct extraction service based on the classified image type.

    Routing table
    ─────────────
    TEXT_IMAGE    → extract_text_knowledge()
    CHART         → extract_chart_knowledge()
    DIAGRAM       → extract_diagram_knowledge()
    MIXED         → run_visual_understanding_logic()
    NATURAL_IMAGE → describe_image_with_gemini()
    UNKNOWN       → no extractor (return descriptive payload)
    """
    section("STEP 2 — KNOWLEDGE EXTRACTION")

    ext = os.path.splitext(image_path)[1].lower()
    extractor_name = ""
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    if image_type == "TEXT_IMAGE":
        from RAG.services.text_knowledge_extractor import extract_text_knowledge

        extractor_name = "extract_text_knowledge"
        logger.info("Routing to %s | mime_type=%s", extractor_name, mime_type)
        print(f"\n  Extractor Selected : {extractor_name}")

        result = extract_text_knowledge(image_bytes, mime_type)

    # ------------------------------------------------------------------ #
    elif image_type == "CHART":
        from RAG.services.chart_knowledge_extractor import extract_chart_knowledge

        extractor_name = "extract_chart_knowledge"
        logger.info("Routing to %s | mime_type=%s", extractor_name, mime_type)
        print(f"\n  Extractor Selected : {extractor_name}")

        result = extract_chart_knowledge(image_bytes, mime_type)

    # ------------------------------------------------------------------ #
    elif image_type == "DIAGRAM":
        from RAG.services.diagram_knowledge_extractor import extract_diagram_knowledge

        extractor_name = "extract_diagram_knowledge"
        logger.info("Routing to %s | mime_type=%s", extractor_name, mime_type)
        print(f"\n  Extractor Selected : {extractor_name}")

        result = extract_diagram_knowledge(image_bytes, mime_type)

    # ------------------------------------------------------------------ #
    elif image_type == "MIXED":
        from RAG.services.image_intelligence import run_visual_understanding_logic

        extractor_name = "run_visual_understanding_logic"
        logger.info("Routing to %s | ext=%s", extractor_name, ext)
        print(f"\n  Extractor Selected : {extractor_name}")

        raw = run_visual_understanding_logic(image_path, ext)

        # Normalise to the standard output shape used by the other extractors
        result = {
            "image_type": "MIXED",
            "ocr_text": raw.get("ocr_text", ""),
            "vision_summary": raw.get("vision_summary", ""),
            "combined_understanding": raw.get("combined_understanding", ""),
            # Provide rich_text_representation so the display section is uniform
            "rich_text_representation": (
                f"# Mixed Image Analysis\n\n"
                f"## OCR Text:\n{raw.get('ocr_text', '[None]')}\n\n"
                f"## Vision Summary:\n{raw.get('vision_summary', '[None]')}\n\n"
                f"## Combined Understanding:\n{raw.get('combined_understanding', '[None]')}\n"
            )
        }

    # ------------------------------------------------------------------ #
    elif image_type == "NATURAL_IMAGE":
        from RAG.ingestion import describe_image_with_gemini

        extractor_name = "describe_image_with_gemini"
        logger.info("Routing to %s | mime_type=%s", extractor_name, mime_type)
        print(f"\n  Extractor Selected : {extractor_name}")

        summary = describe_image_with_gemini(image_bytes, mime_type)

        # Wrap the plain string into the standard output shape
        result = {
            "image_type": "NATURAL_IMAGE",
            "summary": summary,
            "rich_text_representation": (
                f"# Natural Image Description\n\n"
                f"## Gemini Vision Summary:\n{summary}\n"
            )
        }

    # ------------------------------------------------------------------ #
    else:
        # UNKNOWN — no extractor is appropriate
        extractor_name = "none (UNKNOWN classification)"
        logger.warning("image_type=UNKNOWN — skipping knowledge extraction.")
        print(f"\n  Extractor Selected : {extractor_name}")

        result = {
            "image_type": "UNKNOWN",
            "rich_text_representation": (
                "# Unknown Image\n\n"
                "Classification confidence was too low to select a knowledge extractor.\n"
            )
        }

    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - t0
    logger.info(
        "Extraction complete | extractor=%s | elapsed=%s",
        extractor_name,
        elapsed_label(elapsed)
    )
    print(f"  Elapsed            : {elapsed_label(elapsed)}")

    result["_extractor_name"] = extractor_name
    result["_elapsed_extraction"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Step 3 — Pretty Print Results
# ---------------------------------------------------------------------------

def display_results(classification: dict, extraction: dict):
    """Formats and prints the complete pipeline output to stdout."""

    section("PIPELINE RESULTS")

    # --- Classification summary ---
    safe_print("\n  ┌─ Classification")
    safe_print(f"  │  Image Type   : {classification['image_type']}")
    safe_print(f"  │  Confidence   : {classification['confidence']:.4f}")
    safe_print(f"  │  Reason       : {classification['reason']}")
    safe_print(f"  └─ Elapsed      : {elapsed_label(classification['_elapsed_classification'])}")

    # --- Extraction summary ---
    safe_print("\n  ┌─ Extraction")
    safe_print(f"  │  Extractor    : {extraction['_extractor_name']}")
    safe_print(f"  └─ Elapsed      : {elapsed_label(extraction['_elapsed_extraction'])}")

    # --- Full JSON knowledge object ---
    section("EXTRACTED KNOWLEDGE OBJECT (JSON)")
    # Remove internal timing keys before display
    display_dict = {k: v for k, v in extraction.items() if not k.startswith("_")}
    safe_print(json.dumps(display_dict, indent=2, ensure_ascii=False, default=str))

    # --- Rich text representation ---
    rich_text = extraction.get("rich_text_representation", "")
    if rich_text:
        section("RICH TEXT REPRESENTATION")
        safe_print(rich_text)


# ---------------------------------------------------------------------------
# Step 4 — Pipeline timing summary
# ---------------------------------------------------------------------------

def display_timing_summary(classification: dict, extraction: dict, total_elapsed: float):
    """Prints a timing breakdown for all pipeline stages."""
    section("TIMING SUMMARY")
    t_cls = classification["_elapsed_classification"]
    t_ext = extraction["_elapsed_extraction"]
    t_ovr = total_elapsed - t_cls - t_ext   # overhead (IO, imports, etc.)

    rows = [
        ("Classification", t_cls),
        ("Extraction",     t_ext),
        ("I/O & Overhead", t_ovr),
        ("Total Pipeline", total_elapsed),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    for label, secs in rows:
        bar = "█" * max(1, int(secs * 20))
        safe_print(f"  {label:<{col_w}} {elapsed_label(secs):>8}   {bar}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Image Pipeline Validation Script\n\n"
            "Classifies an image, routes it to the correct knowledge extractor,\n"
            "and pretty-prints the full extraction result with timing logs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image file to process (.png, .jpg, .jpeg, .webp)"
    )
    parser.add_argument(
        "--path",
        dest="image_path_flag",
        default=None,
        help="Alternative flag-style image path argument: --path /path/to/image.png"
    )
    parser.add_argument(
        "--save",
        dest="save_output",
        default=None,
        metavar="OUTPUT_PATH",
        help="Optional: save the JSON extraction result to this file path."
    )
    args = parser.parse_args()

    # Resolve image path from positional or --path flag
    image_path = args.image_path or args.image_path_flag

    # Auto-discover fallback: look in outputs/images/ first, then project root
    if not image_path:
        search_dirs = [
            os.path.join(BASE_DIR, "outputs", "images"),
            BASE_DIR,
        ]
        exts = (".png", ".jpg", ".jpeg", ".webp")
        for d in search_dirs:
            if os.path.isdir(d):
                hits = [f for f in os.listdir(d) if f.lower().endswith(exts)]
                if hits:
                    image_path = os.path.join(d, hits[0])
                    break

    if not image_path:
        print(
            "[ERROR] No image file specified and none found automatically.\n"
            "        Usage: python RAG/tests/test_image_pipeline.py <image_path>\n"
            "           or: python RAG/tests/test_image_pipeline.py --path <image_path>"
        )
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    # Guard: GEMINI_API_KEY must be set
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "[ERROR] GEMINI_API_KEY is not set.\n"
            "        Add it to your .env file or set it in your environment."
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # Print header
    # ----------------------------------------------------------------
    mime_type = get_mime_type(image_path)
    file_size = os.path.getsize(image_path)

    section("IMAGE PIPELINE VALIDATION")
    print(f"\n  Image Path : {image_path}")
    print(f"  MIME Type  : {mime_type}")
    print(f"  File Size  : {file_size:,} bytes")

    logger.info(
        "Pipeline started | image=%s | mime_type=%s | size=%d bytes",
        os.path.basename(image_path),
        mime_type,
        file_size
    )

    # ----------------------------------------------------------------
    # Read image bytes once — shared across both steps
    # ----------------------------------------------------------------
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logger.error("Failed to read image file: %s", e, exc_info=True)
        print(f"[ERROR] Could not read image file: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Execute pipeline
    # ----------------------------------------------------------------
    pipeline_start = time.perf_counter()

    try:
        # Step 1 — Classify
        classification = run_classification(image_bytes, mime_type)
    except Exception as e:
        logger.error("Classification step failed: %s", e, exc_info=True)
        print(f"\n[ERROR] Classification failed: {e}")
        sys.exit(1)

    try:
        # Step 2 — Route and extract
        extraction = route_and_extract(
            image_type=classification["image_type"],
            image_bytes=image_bytes,
            mime_type=mime_type,
            image_path=image_path
        )
    except Exception as e:
        logger.error(
            "Extraction step failed | extractor routing for image_type=%s | error=%s",
            classification.get("image_type", "?"),
            e,
            exc_info=True
        )
        print(f"\n[ERROR] Extraction failed: {e}")
        sys.exit(1)

    total_elapsed = time.perf_counter() - pipeline_start

    # ----------------------------------------------------------------
    # Display results
    # ----------------------------------------------------------------
    display_results(classification, extraction)
    display_timing_summary(classification, extraction, total_elapsed)

    # ----------------------------------------------------------------
    # Optional: Save JSON output to file
    # ----------------------------------------------------------------
    if args.save_output:
        save_path = args.save_output
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        output_data = {
            "image_path": image_path,
            "mime_type": mime_type,
            "classification": {
                "image_type": classification["image_type"],
                "confidence": classification["confidence"],
                "reason": classification["reason"],
                "elapsed_seconds": round(classification["_elapsed_classification"], 4)
            },
            "extraction": {
                k: v for k, v in extraction.items() if not k.startswith("_")
            },
            "timing": {
                "classification_seconds": round(classification["_elapsed_classification"], 4),
                "extraction_seconds": round(extraction["_elapsed_extraction"], 4),
                "total_seconds": round(total_elapsed, 4)
            }
        }
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            section("OUTPUT SAVED")
            print(f"\n  JSON result written to: {save_path}")
            logger.info("Pipeline output saved to: %s", save_path)
        except Exception as e:
            logger.error("Failed to save output file: %s", e, exc_info=True)
            print(f"\n[WARN] Could not save output file: {e}")

    section("PIPELINE COMPLETE")
    logger.info(
        "Pipeline finished | image_type=%s | total_elapsed=%s",
        classification["image_type"],
        elapsed_label(total_elapsed)
    )
    print()


if __name__ == "__main__":
    main()
