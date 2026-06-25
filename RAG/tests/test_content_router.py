import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the root folder is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from RAG.content_router import route_content, RoutingMetadata

class TestContentRouter(unittest.TestCase):
    def test_static_routing_by_name(self):
        # 1. PDF
        meta = route_content(file_name="sample_doc.pdf")
        self.assertEqual(meta.file_name, "sample_doc.pdf")
        self.assertEqual(meta.file_extension, "pdf")
        self.assertEqual(meta.mime_type, "application/pdf")
        self.assertEqual(meta.primary_category, "document")
        self.assertTrue(meta.is_supported)
        self.assertEqual(meta.suggested_route, "pdf_pipeline")
        self.assertEqual(meta.suggested_extractor, "process_pdf")
        self.assertIsNone(meta.classification)

        # 2. PNG Image
        meta = route_content(file_name="visual.png")
        self.assertEqual(meta.file_name, "visual.png")
        self.assertEqual(meta.file_extension, "png")
        self.assertEqual(meta.mime_type, "image/png")
        self.assertEqual(meta.primary_category, "image")
        self.assertTrue(meta.is_supported)
        self.assertEqual(meta.suggested_route, "image_pipeline")
        self.assertEqual(meta.suggested_extractor, "classify_image")

        # 3. CSV Table
        meta = route_content(file_name="metrics.csv")
        self.assertEqual(meta.file_name, "metrics.csv")
        self.assertEqual(meta.file_extension, "csv")
        self.assertEqual(meta.mime_type, "text/csv")
        self.assertEqual(meta.primary_category, "spreadsheet")
        self.assertTrue(meta.is_supported)
        self.assertEqual(meta.suggested_route, "table_pipeline")
        self.assertEqual(meta.suggested_extractor, "classify_table")

    def test_static_routing_by_path(self):
        meta = route_content(file_path="C:/some/dir/report.xlsx")
        self.assertEqual(meta.file_name, "report.xlsx")
        self.assertEqual(meta.file_extension, "xlsx")
        self.assertEqual(meta.mime_type, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.assertEqual(meta.primary_category, "spreadsheet")
        self.assertTrue(meta.is_supported)
        self.assertEqual(meta.suggested_route, "table_pipeline")
        self.assertEqual(meta.suggested_extractor, "classify_table")

    def test_static_routing_by_mime_type(self):
        # Reverse mapping xls
        meta = route_content(mime_type="application/vnd.ms-excel")
        self.assertEqual(meta.file_name, "unknown_file.xls")
        self.assertEqual(meta.file_extension, "xls")
        self.assertEqual(meta.mime_type, "application/vnd.ms-excel")
        self.assertEqual(meta.primary_category, "spreadsheet")
        self.assertTrue(meta.is_supported)

    def test_unsupported_file_formats(self):
        # Text file
        meta = route_content(file_name="notes.txt")
        self.assertEqual(meta.file_extension, "txt")
        self.assertFalse(meta.is_supported)
        self.assertEqual(meta.primary_category, "unknown")
        self.assertEqual(meta.suggested_route, "unknown_pipeline")
        self.assertEqual(meta.suggested_extractor, "none")
        self.assertIsNone(meta.classification)

    def test_missing_all_parameters_raises_value_error(self):
        with self.assertRaises(ValueError):
            route_content()

    @patch("RAG.services.image_classifier.classify_image")
    def test_dynamic_routing_image_with_bytes(self, mock_classify_image):
        mock_classify_image.return_value = {
            "image_type": "DIAGRAM",
            "confidence": 0.92,
            "reason": "Flowchart representing architecture logic."
        }

        meta = route_content(
            file_name="flowchart.jpg",
            file_bytes=b"mock_image_binary_data"
        )

        self.assertEqual(meta.file_extension, "jpg")
        self.assertEqual(meta.primary_category, "image")
        self.assertEqual(meta.suggested_extractor, "extract_diagram_knowledge")
        self.assertIsNotNone(meta.classification)
        self.assertEqual(meta.classification["type"], "DIAGRAM")
        self.assertEqual(meta.classification["confidence"], 0.92)
        self.assertEqual(meta.classification["reason"], "Flowchart representing architecture logic.")
        mock_classify_image.assert_called_once_with(b"mock_image_binary_data", "image/jpeg")

    @patch("RAG.services.table_classifier.classify_table")
    @patch("RAG.routes.validation.parse_table_file_to_markdown")
    def test_dynamic_routing_table_with_bytes(self, mock_parse_table, mock_classify_table):
        mock_parse_table.return_value = "| Year | Revenue |\n|---|---|\n| 2024 | $10M |"
        mock_classify_table.return_value = {
            "table_type": "TABLE_FINANCIAL",
            "confidence": 0.98,
            "reason": "Contains annual revenue timeline."
        }

        meta = route_content(
            file_name="revenue.xlsx",
            file_bytes=b"mock_excel_binary_data"
        )

        self.assertEqual(meta.file_extension, "xlsx")
        self.assertEqual(meta.primary_category, "spreadsheet")
        self.assertEqual(meta.suggested_extractor, "extract_financial_table")
        self.assertIsNotNone(meta.classification)
        self.assertEqual(meta.classification["type"], "TABLE_FINANCIAL")
        self.assertEqual(meta.classification["confidence"], 0.98)
        self.assertEqual(meta.classification["reason"], "Contains annual revenue timeline.")
        mock_parse_table.assert_called_once()
        mock_classify_table.assert_called_once_with(mock_parse_table.return_value)

if __name__ == "__main__":
    unittest.main()
