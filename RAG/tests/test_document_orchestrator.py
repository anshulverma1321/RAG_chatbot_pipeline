"""
RAG/tests/test_document_orchestrator.py

Unit tests for PDF document orchestrator.
Mocks PDF parsing (pdfplumber, pypdf) and asset routing (classifiers, extractors)
to verify correct multi-page knowledge chunking and output normalization.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from RAG.document_orchestrator import orchestrate_pdf
from RAG.knowledge_normalizer import NormalizedKnowledgeChunk

class TestDocumentOrchestrator(unittest.TestCase):
    """Test PDF page-by-page orchestration, layout parsing, and routing logic with normalization."""

    @patch("RAG.document_orchestrator.PdfReader")
    @patch("RAG.document_orchestrator.pdfplumber.open")
    @patch("RAG.document_orchestrator.classify_image")
    @patch("RAG.document_orchestrator.classify_table")
    @patch("RAG.services.chart_knowledge_extractor.extract_chart_knowledge")
    @patch("RAG.document_orchestrator.extract_financial_table")
    @patch("RAG.document_orchestrator.split_text")
    def test_pdf_orchestration_flow(
        self,
        mock_split,
        mock_extract_financial,
        mock_extract_chart,
        mock_classify_table,
        mock_classify_image,
        mock_plumber_open,
        mock_pypdf_reader
    ):
        # 1. Mock file existence check
        with patch("RAG.document_orchestrator.os.path.exists", return_value=True):
            # 2. Mock PdfReader (pypdf) page and images
            mock_img = MagicMock()
            mock_img.name = "chart.png"
            mock_img.data = b"fakeimagebytes"
            
            mock_pypdf_page = MagicMock()
            mock_pypdf_page.images = [mock_img]
            
            mock_reader_instance = MagicMock()
            mock_reader_instance.pages = [mock_pypdf_page]
            mock_pypdf_reader.return_value = mock_reader_instance

            # 3. Mock pdfplumber page, tables and text
            mock_plumber_page = MagicMock()
            mock_plumber_page.extract_text.return_value = "Page text paragraph."
            # Table layout structure: 2x2 table
            mock_plumber_page.extract_tables.return_value = [
                [["Header 1", "Header 2"], ["Val 1", "Val 2"]]
            ]
            
            mock_plumber_instance = MagicMock()
            mock_plumber_instance.pages = [mock_plumber_page]
            mock_plumber_open.return_value = mock_plumber_instance

            # 4. Mock classifiers
            mock_classify_image.return_value = {"image_type": "CHART", "confidence": 0.9}
            mock_classify_table.return_value = {"table_type": "TABLE_FINANCIAL", "confidence": 0.95}

            # 5. Mock extractors
            mock_extract_chart.return_value = {
                "chart_type": "bar",
                "title": "Chart Title",
                "rich_text_representation": "# Chart Extraction\nMock markdown"
            }
            mock_extract_financial.return_value = {
                "title": "Financial Sheet",
                "currency": "USD",
                "rich_text_representation": "# Financial Report\nSummary"
            }
            mock_split.return_value = ["Page text paragraph."]

            # 6. Execute orchestrator
            chunks = orchestrate_pdf("mock_document.pdf")

            # 7. Assertions
            self.assertEqual(len(chunks), 3)  # 1 table chunk, 1 image chunk, 1 text chunk
            
            # Verify that all chunks are NormalizedKnowledgeChunk instances
            for c in chunks:
                self.assertIsInstance(c, NormalizedKnowledgeChunk)
                self.assertIsNotNone(c.chunk_id)
                self.assertEqual(c.document_name, "mock_document.pdf")
                self.assertEqual(c.metadata["document_name"], "mock_document.pdf")

            # Verify table chunk
            table_chunk = next(c for c in chunks if c.asset_type == "table")
            self.assertEqual(table_chunk.page_number, 1)
            self.assertEqual(table_chunk.classification_type, "TABLE_FINANCIAL")
            self.assertEqual(table_chunk.extractor_used, "extract_financial_table")
            self.assertEqual(table_chunk.rich_text_representation, "# Financial Report\nSummary")
            self.assertEqual(table_chunk.structured_knowledge, {"title": "Financial Sheet", "currency": "USD"})
            self.assertIn("Financial Sheet", table_chunk.embedding_text)

            # Verify image chunk
            image_chunk = next(c for c in chunks if c.asset_type == "image")
            self.assertEqual(image_chunk.page_number, 1)
            self.assertEqual(image_chunk.classification_type, "CHART")
            self.assertEqual(image_chunk.extractor_used, "extract_chart_knowledge")
            self.assertEqual(image_chunk.rich_text_representation, "# Chart Extraction\nMock markdown")
            self.assertEqual(image_chunk.structured_knowledge, {"chart_type": "bar", "title": "Chart Title"})
            self.assertIn("Chart Title", image_chunk.embedding_text)

            # Verify text chunk
            text_chunk = next(c for c in chunks if c.asset_type == "text")
            self.assertEqual(text_chunk.page_number, 1)
            self.assertEqual(text_chunk.classification_type, "text")
            self.assertEqual(text_chunk.extractor_used, "split_text")
            self.assertEqual(text_chunk.rich_text_representation, "Page text paragraph.")
            self.assertEqual(text_chunk.structured_knowledge, {"text": "Page text paragraph."})
            self.assertEqual(text_chunk.embedding_text, "Page text paragraph.")

            # Check that mocks were correctly invoked
            mock_plumber_open.assert_called_once_with("mock_document.pdf")
            mock_pypdf_reader.assert_called_once_with("mock_document.pdf")
            mock_classify_image.assert_called_once_with(b"fakeimagebytes", "image/png")
            mock_classify_table.assert_called_once()
            mock_extract_chart.assert_called_once_with(b"fakeimagebytes", "image/png")
            mock_extract_financial.assert_called_once()
            mock_split.assert_called_once_with("Page text paragraph.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
