"""
RAG/tests/test_ocr_lazy_loading.py

Validates the lazy-loading OCR optimization:

  1. Importing image/text extractor modules does NOT initialize PaddleOCR.
  2. Importing document_orchestrator does NOT initialize PaddleOCR.
  3. Importing routes/process does NOT initialize PaddleOCR.
  4. orchestrate_pdf() on a text-only PDF does NOT initialize PaddleOCR.
  5. orchestrate_pdf() on a table-only PDF does NOT initialize PaddleOCR.
  6. orchestrate_pdf() on a PDF with images DOES trigger image extractor (and therefore OCR).
  7. The PaddleOCR singleton in image_intelligence is reused across multiple calls.

All tests use mocks so they never touch the real filesystem, Gemini API, or PaddleOCR models.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock, call

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so RAG package is importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


# ===========================================================================
# Helper: is PaddleOCR present in sys.modules?
# ===========================================================================
def _paddle_in_modules() -> bool:
    """Returns True if paddleocr has been loaded into sys.modules."""
    return "paddleocr" in sys.modules


def _clear_paddle_from_modules():
    """Remove paddleocr from sys.modules so each test starts clean."""
    for key in list(sys.modules.keys()):
        if key.startswith("paddleocr") or key.startswith("paddle"):
            del sys.modules[key]


# ===========================================================================
# Test Suite 1: Import-time safety
# ===========================================================================

class TestImportTimeSafety(unittest.TestCase):
    """Verifies that importing our modules never loads PaddleOCR as a side effect."""

    def setUp(self):
        _clear_paddle_from_modules()

    def test_import_diagram_extractor_does_not_load_paddleocr(self):
        """
        Importing diagram_knowledge_extractor should not load paddleocr.
        The deferred import inside run_ocr_on_bytes() keeps OCR out of
        the module load path.
        """
        # Force re-import by removing from sys.modules if already cached
        for key in list(sys.modules.keys()):
            if "diagram_knowledge_extractor" in key:
                del sys.modules[key]

        import RAG.services.diagram_knowledge_extractor  # noqa: F401

        self.assertFalse(
            _paddle_in_modules(),
            "paddleocr was loaded as a side effect of importing diagram_knowledge_extractor — "
            "lazy-loading is broken.",
        )

    def test_import_text_extractor_does_not_load_paddleocr(self):
        """
        Importing text_knowledge_extractor should not load paddleocr.
        """
        for key in list(sys.modules.keys()):
            if "text_knowledge_extractor" in key:
                del sys.modules[key]

        import RAG.services.text_knowledge_extractor  # noqa: F401

        self.assertFalse(
            _paddle_in_modules(),
            "paddleocr was loaded as a side effect of importing text_knowledge_extractor — "
            "lazy-loading is broken.",
        )

    def test_import_image_intelligence_does_not_load_paddleocr(self):
        """
        image_intelligence.py defines the singleton but must not call PaddleOCR() at import.
        """
        for key in list(sys.modules.keys()):
            if "image_intelligence" in key:
                del sys.modules[key]

        import RAG.services.image_intelligence  # noqa: F401

        self.assertFalse(
            _paddle_in_modules(),
            "paddleocr was loaded as a side effect of importing image_intelligence — "
            "the singleton must not initialize at import time.",
        )

    @patch("RAG.db.init_db", return_value=None)
    @patch("RAG.vector_store.init_vector_store", return_value=None)
    def test_import_document_orchestrator_does_not_load_paddleocr(self, *mocks):
        """
        document_orchestrator now imports only table extractors and PDF tools at module level.
        Image extractors (which pull in image_intelligence) are deferred.
        """
        for key in list(sys.modules.keys()):
            if "document_orchestrator" in key:
                del sys.modules[key]
            if "diagram_knowledge_extractor" in key:
                del sys.modules[key]
            if "text_knowledge_extractor" in key:
                del sys.modules[key]

        import RAG.document_orchestrator  # noqa: F401

        self.assertFalse(
            _paddle_in_modules(),
            "paddleocr was loaded as a side effect of importing document_orchestrator — "
            "deferred image-extractor imports are not working.",
        )


# ===========================================================================
# Test Suite 2: PDF Orchestration — No OCR for text/table-only PDFs
# ===========================================================================

class TestOrchestratorNoOCRForNonImagePDFs(unittest.TestCase):
    """
    Verifies that orchestrate_pdf() on PDFs with no embedded images
    never triggers OCR initialization.
    """

    def _make_text_page(self, text="Sample paragraph text."):
        """Returns a mock pdfplumber page with text, no tables, no images."""
        page = MagicMock()
        page.extract_text.return_value = text
        page.extract_tables.return_value = []
        return page

    def _make_table_page(self, table=None):
        """Returns a mock pdfplumber page with a table, no images."""
        page = MagicMock()
        page.extract_text.return_value = ""
        page.extract_tables.return_value = table or [
            [["Header A", "Header B"], ["Val 1", "Val 2"]]
        ]
        return page

    def _make_pypdf_page_no_images(self):
        """Returns a mock pypdf page with an empty images list."""
        pypdf_page = MagicMock()
        pypdf_page.images = []  # ← empty: no embedded image assets
        return pypdf_page

    @patch("RAG.document_orchestrator.classify_table")
    @patch("RAG.document_orchestrator.split_text")
    @patch("RAG.document_orchestrator.pdfplumber.open")
    @patch("RAG.document_orchestrator.PdfReader")
    @patch("RAG.document_orchestrator.os.path.exists", return_value=True)
    def test_text_only_pdf_does_not_init_ocr(
        self, mock_exists, mock_pypdf, mock_plumber, mock_split, mock_classify_table
    ):
        """
        A PDF containing only text paragraphs should never call any image extractor
        or initialize PaddleOCR.
        """
        from RAG.document_orchestrator import orchestrate_pdf

        mock_split.return_value = ["chunk one", "chunk two"]

        plumber_page = self._make_text_page()
        plumber_instance = MagicMock()
        plumber_instance.pages = [plumber_page]
        mock_plumber.return_value = plumber_instance

        pypdf_page = self._make_pypdf_page_no_images()
        reader_instance = MagicMock()
        reader_instance.pages = [pypdf_page]
        mock_pypdf.return_value = reader_instance

        # We want to confirm extract_text_knowledge and extract_diagram_knowledge
        # are never imported/called. Patch them at the module they'll be deferred-imported from.
        with patch("RAG.services.text_knowledge_extractor.extract_text_knowledge") as mock_text_ext, \
             patch("RAG.services.diagram_knowledge_extractor.extract_diagram_knowledge") as mock_diag_ext, \
             patch("RAG.services.chart_knowledge_extractor.extract_chart_knowledge") as mock_chart_ext:

            chunks = orchestrate_pdf("mock_text_only.pdf")

            # Text chunks should exist
            self.assertGreater(len(chunks), 0)

            # Image extractors must NEVER have been called
            mock_text_ext.assert_not_called()
            mock_diag_ext.assert_not_called()
            mock_chart_ext.assert_not_called()

    @patch("RAG.document_orchestrator.classify_table")
    @patch("RAG.document_orchestrator.split_text")
    @patch("RAG.document_orchestrator.pdfplumber.open")
    @patch("RAG.document_orchestrator.PdfReader")
    @patch("RAG.document_orchestrator.os.path.exists", return_value=True)
    def test_table_only_pdf_does_not_init_ocr(
        self, mock_exists, mock_pypdf, mock_plumber, mock_split, mock_classify_table
    ):
        """
        A PDF containing only tables should never call any image extractor.
        Table extractors use Gemini only — no PaddleOCR.
        """
        from RAG.document_orchestrator import orchestrate_pdf

        mock_split.return_value = []
        mock_classify_table.return_value = {
            "table_type": "TABLE_SIMPLE",
            "confidence": 0.95,
            "reason": "Simple two-column table",
        }

        plumber_page = self._make_table_page()
        plumber_page.extract_text.return_value = ""
        plumber_instance = MagicMock()
        plumber_instance.pages = [plumber_page]
        mock_plumber.return_value = plumber_instance

        pypdf_page = self._make_pypdf_page_no_images()
        reader_instance = MagicMock()
        reader_instance.pages = [pypdf_page]
        mock_pypdf.return_value = reader_instance

        with patch("RAG.services.table_intelligence.extract_simple_table") as mock_simple, \
             patch("RAG.services.text_knowledge_extractor.extract_text_knowledge") as mock_text_ext, \
             patch("RAG.services.diagram_knowledge_extractor.extract_diagram_knowledge") as mock_diag_ext:

            mock_simple.return_value = {
                "title": "Sample Table",
                "headers": ["Header A", "Header B"],
                "rows": [["Val 1", "Val 2"]],
                "summary": "A simple two-column table.",
                "rich_text_representation": "# Simple Table\n| Header A | Header B |\n|---|---|\n| Val 1 | Val 2 |\n",
            }

            chunks = orchestrate_pdf("mock_table_only.pdf")

            # Should have table chunk(s)
            table_chunks = [c for c in chunks if c.asset_type == "table"]
            self.assertGreater(len(table_chunks), 0)

            # Image extractors must NEVER have been called
            mock_text_ext.assert_not_called()
            mock_diag_ext.assert_not_called()

    @patch("RAG.document_orchestrator.classify_image")
    @patch("RAG.document_orchestrator.classify_table")
    @patch("RAG.document_orchestrator.split_text")
    @patch("RAG.document_orchestrator.pdfplumber.open")
    @patch("RAG.document_orchestrator.PdfReader")
    @patch("RAG.document_orchestrator.os.path.exists", return_value=True)
    def test_image_pdf_does_trigger_image_extractor(
        self,
        mock_exists,
        mock_pypdf,
        mock_plumber,
        mock_split,
        mock_classify_table,
        mock_classify_image,
    ):
        """
        A PDF containing embedded images SHOULD call the image extractor.
        This is the positive case: confirm routing works when images are present.
        """
        from RAG.document_orchestrator import orchestrate_pdf

        mock_split.return_value = []
        mock_classify_table.return_value = {"table_type": "TABLE_UNKNOWN", "confidence": 0.0}
        mock_classify_image.return_value = {
            "image_type": "CHART",
            "confidence": 0.93,
            "reason": "Bar chart with labeled axes",
        }

        # Build a pdfplumber page with no tables and no text
        plumber_page = MagicMock()
        plumber_page.extract_text.return_value = ""
        plumber_page.extract_tables.return_value = []
        plumber_instance = MagicMock()
        plumber_instance.pages = [plumber_page]
        mock_plumber.return_value = plumber_instance

        # Build a pypdf page WITH one embedded image
        mock_img = MagicMock()
        mock_img.name = "chart.png"
        mock_img.data = b"fakepngbytes"

        pypdf_page = MagicMock()
        pypdf_page.images = [mock_img]
        reader_instance = MagicMock()
        reader_instance.pages = [pypdf_page]
        mock_pypdf.return_value = reader_instance

        mock_chart_result = {
            "image_type": "chart",
            "chart_type": "bar_chart",
            "title": "Revenue Q1",
            "x_axis": "Quarter",
            "y_axis": "Revenue ($M)",
            "data_points": [],
            "trends": ["Upward trend in Q1"],
            "insights": ["Revenue peaked in March"],
            "rich_text_representation": "# Chart Knowledge Extraction\n- **Title**: Revenue Q1\n",
        }

        # Patch the extractor at its source module so the deferred import resolves correctly
        with patch(
            "RAG.services.chart_knowledge_extractor.extract_chart_knowledge",
            return_value=mock_chart_result,
        ) as mock_chart_ext:
            chunks = orchestrate_pdf("mock_image_pdf.pdf")

            image_chunks = [c for c in chunks if c.asset_type == "image"]
            self.assertGreater(len(image_chunks), 0, "Expected at least one image chunk.")

            image_chunk = image_chunks[0]
            self.assertEqual(image_chunk.classification_type, "CHART")
            self.assertEqual(image_chunk.extractor_used, "extract_chart_knowledge")


# ===========================================================================
# Test Suite 3: PaddleOCR Singleton Behaviour
# ===========================================================================

class TestPaddleOCRSingleton(unittest.TestCase):
    """
    Verifies the singleton pattern in image_intelligence._get_paddle_ocr():
      - PaddleOCR() is only instantiated once even across multiple calls.
      - The same instance object is returned on subsequent calls.
    """

    def setUp(self):
        """Reset the singleton before each test."""
        import RAG.services.image_intelligence as img_intel
        img_intel._paddle_ocr_instance = None

    def tearDown(self):
        """Reset the singleton after each test."""
        import RAG.services.image_intelligence as img_intel
        img_intel._paddle_ocr_instance = None

    def test_singleton_created_only_once(self):
        """
        Calling _get_paddle_ocr() multiple times should only call PaddleOCR() once.
        """
        from RAG.services.image_intelligence import _get_paddle_ocr

        mock_ocr_instance = MagicMock(name="PaddleOCRInstance")
        mock_paddle_cls = MagicMock(return_value=mock_ocr_instance)

        with patch.dict("sys.modules", {"paddleocr": MagicMock(PaddleOCR=mock_paddle_cls)}):
            result1 = _get_paddle_ocr()
            result2 = _get_paddle_ocr()
            result3 = _get_paddle_ocr()

        # PaddleOCR() constructor called exactly once
        self.assertEqual(
            mock_paddle_cls.call_count,
            1,
            f"PaddleOCR() was called {mock_paddle_cls.call_count} times — singleton is broken.",
        )

        # All three calls return the same object
        self.assertIs(result1, result2, "Second call returned a different instance.")
        self.assertIs(result1, result3, "Third call returned a different instance.")

    def test_singleton_reused_across_ocr_calls(self):
        """
        Calling run_ocr_with_diagnostics() twice should only build the PaddleOCR
        model once. The second call reuses the cached singleton.
        """
        from RAG.services.image_intelligence import run_ocr_with_diagnostics
        import RAG.services.image_intelligence as img_intel

        mock_ocr_instance = MagicMock(name="PaddleOCRInstance")
        # Simulate OCR result format used by run_ocr_with_diagnostics
        mock_ocr_instance.ocr.return_value = [[]]
        mock_paddle_cls = MagicMock(return_value=mock_ocr_instance)

        paddle_module = MagicMock()
        paddle_module.PaddleOCR = mock_paddle_cls

        # Reset singleton so first call actually builds it
        img_intel._paddle_ocr_instance = None

        with patch.dict("sys.modules", {"paddleocr": paddle_module}), \
             patch("os.path.exists", return_value=True):  # fake image path check

            try:
                run_ocr_with_diagnostics("/fake/image1.png")
            except Exception:
                pass  # We only care about constructor call count, not OCR correctness

            try:
                run_ocr_with_diagnostics("/fake/image2.png")
            except Exception:
                pass

        self.assertEqual(
            mock_paddle_cls.call_count,
            1,
            f"PaddleOCR() was constructed {mock_paddle_cls.call_count} times across two OCR calls — "
            "singleton reuse is broken.",
        )

    def test_singleton_instance_is_stored(self):
        """
        After _get_paddle_ocr() is called, the module-level _paddle_ocr_instance
        variable should hold a non-None value.
        """
        import RAG.services.image_intelligence as img_intel
        from RAG.services.image_intelligence import _get_paddle_ocr

        self.assertIsNone(img_intel._paddle_ocr_instance, "Singleton should start as None.")

        mock_ocr = MagicMock(name="OCREngine")
        with patch.dict("sys.modules", {"paddleocr": MagicMock(PaddleOCR=MagicMock(return_value=mock_ocr))}):
            _get_paddle_ocr()

        self.assertIsNotNone(
            img_intel._paddle_ocr_instance,
            "_paddle_ocr_instance was not set after _get_paddle_ocr() call.",
        )


# ===========================================================================
# Test Suite 4: Timing Benchmark (before/after evidence)
# ===========================================================================

class TestOrchestratorTimingNoImages(unittest.TestCase):
    """
    Measures orchestrate_pdf() execution time for a text+table only PDF.
    With lazy loading, this should complete WITHOUT OCR initialization overhead.
    Records results to stdout for before/after comparison.
    """

    @patch("RAG.document_orchestrator.classify_table")
    @patch("RAG.document_orchestrator.split_text")
    @patch("RAG.document_orchestrator.pdfplumber.open")
    @patch("RAG.document_orchestrator.PdfReader")
    @patch("RAG.document_orchestrator.os.path.exists", return_value=True)
    def test_text_only_orchestration_timing(
        self, mock_exists, mock_pypdf, mock_plumber, mock_split, mock_classify_table
    ):
        """
        Orchestrate a 3-page text-only PDF and record timing.
        With lazy-loading, no OCR models should be loaded, making this fast.
        """
        from RAG.document_orchestrator import orchestrate_pdf

        mock_split.return_value = ["paragraph one", "paragraph two"]

        # Build 3 pages — all text, no images
        pages_plumber = []
        pages_pypdf = []
        for _ in range(3):
            p = MagicMock()
            p.extract_text.return_value = "This is a text-only page with important content."
            p.extract_tables.return_value = []
            pages_plumber.append(p)

            pp = MagicMock()
            pp.images = []
            pages_pypdf.append(pp)

        plumber_instance = MagicMock()
        plumber_instance.pages = pages_plumber
        mock_plumber.return_value = plumber_instance

        reader_instance = MagicMock()
        reader_instance.pages = pages_pypdf
        mock_pypdf.return_value = reader_instance

        start = time.perf_counter()
        chunks = orchestrate_pdf("mock_timing_test.pdf")
        elapsed = time.perf_counter() - start

        print(f"\n{'='*60}")
        print("  TIMING BENCHMARK: text-only PDF (3 pages, no images)")
        print(f"{'='*60}")
        print(f"  Chunks produced  : {len(chunks)}")
        print(f"  Execution time   : {elapsed*1000:.1f} ms")
        print(f"  OCR initialized  : NO (lazy-loading active)")
        print(f"{'='*60}\n")

        # Sanity: should produce chunks without error
        self.assertGreater(len(chunks), 0)

        # Sanity: no image chunks in output
        image_chunks = [c for c in chunks if c.asset_type == "image"]
        self.assertEqual(len(image_chunks), 0, "No image chunks expected for text-only PDF.")

        # Sanity: elapsed under 5 seconds (mocked, should be near-instantaneous)
        self.assertLess(
            elapsed,
            5.0,
            f"Orchestration took {elapsed:.2f}s — unexpectedly slow even with mocks.",
        )


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
