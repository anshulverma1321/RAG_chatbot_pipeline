"""
RAG/tests/test_process_endpoint.py

Phase 3 — Universal Processing API Test Suite
==============================================
Validates that the POST /process endpoint dynamically routes different file types
to the correct specialized intelligence extractors and handles responses correctly.
"""

import io
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app
from RAG.content_router import RoutingMetadata

_FAKE_PNG = b"\x89PNG\r\n\x1a\n fake png bytes"
_FAKE_CSV = b"Header1,Header2\nValue1,Value2\n"
_FAKE_PDF = b"%PDF-1.4 fake pdf"
_FAKE_TXT = b"unsupported text"


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


class TestUniversalProcessEndpoint(unittest.TestCase):
    """Integration and routing verification tests for POST /process."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    # ------------------------------------------------------------------
    # 1. Chart Image Upload
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.extract_chart_knowledge")
    def test_01_process_chart_image(self, mock_extract, mock_route):
        _sep("TEST 1 — Chart Image routing")
        
        # Configure routing metadata mock
        mock_route.return_value = RoutingMetadata(
            file_name="sales.png",
            file_extension="png",
            mime_type="image/png",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "CHART", "confidence": 0.98, "reason": "Looks like a bar chart."}
        )
        
        # Configure extractor mock
        mock_extract.return_value = {
            "image_type": "chart",
            "chart_type": "bar_chart",
            "title": "Sales Performance",
            "x_axis": "Month",
            "y_axis": "Revenue",
            "data_points": [{"label": "Jan", "value": "120"}],
            "trends": ["Growth trend"],
            "insights": ["Jan peak"],
            "rich_text_representation": "# Chart Extraction\nJan: 120"
        }

        response = self.client.post(
            "/process",
            files={"file": ("sales.png", io.BytesIO(_FAKE_PNG), "image/png")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "extract_chart_knowledge")
        self.assertEqual(data["rich_text_representation"], "# Chart Extraction\nJan: 120")
        self.assertEqual(data["extracted_knowledge"]["chart_type"], "bar_chart")
        self.assertEqual(data["extracted_knowledge"]["title"], "Sales Performance")
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        
        timing = data["execution_timing"]
        self.assertIn("total_seconds", timing)
        self.assertIn("routing_seconds", timing)
        self.assertIn("processing_seconds", timing)

        mock_route.assert_called_once()
        mock_extract.assert_called_once_with(_FAKE_PNG, "image/png")

    # ------------------------------------------------------------------
    # 2. Diagram Image Upload
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.extract_diagram_knowledge")
    def test_02_process_diagram_image(self, mock_extract, mock_route):
        _sep("TEST 2 — Diagram Image routing")

        mock_route.return_value = RoutingMetadata(
            file_name="flow.png",
            file_extension="png",
            mime_type="image/png",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "DIAGRAM", "confidence": 0.95, "reason": "Flowchart detected."}
        )

        mock_extract.return_value = {
            "image_type": "diagram",
            "diagram_type": "flowchart",
            "nodes": ["A", "B"],
            "relationships": [{"from": "A", "to": "B", "label_or_relationship": "next"}],
            "workflow": ["A then B"],
            "components": [],
            "summary": "Simple flowchart explanation.",
            "rich_text_representation": "# Flowchart Topology\nA to B"
        }

        response = self.client.post(
            "/process",
            files={"file": ("flow.png", io.BytesIO(_FAKE_PNG), "image/png")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "extract_diagram_knowledge")
        self.assertEqual(data["rich_text_representation"], "# Flowchart Topology\nA to B")
        self.assertEqual(data["extracted_knowledge"]["nodes"], ["A", "B"])
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        mock_extract.assert_called_once_with(_FAKE_PNG, "image/png")

    # ------------------------------------------------------------------
    # 3. Infographic Image Upload (MIXED)
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.run_visual_understanding_logic_on_bytes")
    def test_03_process_infographic_image(self, mock_extract, mock_route):
        _sep("TEST 3 — Infographic (MIXED) routing")

        mock_route.return_value = RoutingMetadata(
            file_name="info.png",
            file_extension="png",
            mime_type="image/png",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "MIXED", "confidence": 0.91, "reason": "Contains text and diagrams."}
        )

        mock_extract.return_value = {
            "image_type": "MIXED",
            "sections": ["Section 1"],
            "headings": ["Title"],
            "labels": ["Label"],
            "process_flow": [],
            "key_takeaways": ["Takeaway"],
            "summary": "Infographic summary.",
            "rich_text_representation": "# Infographic Analysis\nTakeaway"
        }

        response = self.client.post(
            "/process",
            files={"file": ("info.png", io.BytesIO(_FAKE_PNG), "image/png")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "run_visual_understanding_logic")
        self.assertEqual(data["rich_text_representation"], "# Infographic Analysis\nTakeaway")
        self.assertEqual(data["extracted_knowledge"]["headings"], ["Title"])
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        mock_extract.assert_called_once_with(_FAKE_PNG, "image/png")

    # ------------------------------------------------------------------
    # 4. Text Image Upload
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.extract_text_knowledge")
    def test_04_process_text_image(self, mock_extract, mock_route):
        _sep("TEST 4 — Text Image routing")

        mock_route.return_value = RoutingMetadata(
            file_name="doc.png",
            file_extension="png",
            mime_type="image/png",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "TEXT_IMAGE", "confidence": 0.97, "reason": "Plain text document."}
        )

        mock_extract.return_value = {
            "image_type": "text_image",
            "document_type": "report",
            "extracted_text": "Sample text extracted from OCR.",
            "cleaned_text": "Sample text extracted from OCR.",
            "key_points": ["Point 1"],
            "entities": ["OCR"],
            "word_count": 5,
            "rich_text_representation": "# Document Content\nSample text"
        }

        response = self.client.post(
            "/process",
            files={"file": ("doc.png", io.BytesIO(_FAKE_PNG), "image/png")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "extract_text_knowledge")
        self.assertEqual(data["rich_text_representation"], "# Document Content\nSample text")
        self.assertEqual(data["extracted_knowledge"]["document_type"], "report")
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        mock_extract.assert_called_once_with(_FAKE_PNG, "image/png")

    # ------------------------------------------------------------------
    # 5. Natural Image Upload
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.describe_image_with_gemini")
    def test_05_process_natural_image(self, mock_extract, mock_route):
        _sep("TEST 5 — Natural Image routing")

        mock_route.return_value = RoutingMetadata(
            file_name="photo.jpg",
            file_extension="jpg",
            mime_type="image/jpeg",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "NATURAL_IMAGE", "confidence": 0.94, "reason": "Real-world photograph."}
        )

        mock_extract.return_value = "A photo of a dog sitting in a park."

        response = self.client.post(
            "/process",
            files={"file": ("photo.jpg", io.BytesIO(_FAKE_PNG), "image/jpeg")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "describe_image_with_gemini")
        self.assertIn("Gemini Vision Summary", data["rich_text_representation"])
        self.assertEqual(data["extracted_knowledge"], {"summary": "A photo of a dog sitting in a park."})
        mock_extract.assert_called_once_with(_FAKE_PNG, "image/jpeg")

    # ------------------------------------------------------------------
    # 6. Financial Spreadsheet Upload (XLSX)
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.parse_table_file_to_markdown")
    @patch("RAG.routes.process.extract_financial_table")
    def test_06_process_financial_spreadsheet(self, mock_extract, mock_parse, mock_route):
        _sep("TEST 6 — Financial Spreadsheet routing")

        mock_route.return_value = RoutingMetadata(
            file_name="balance_sheet.xlsx",
            file_extension="xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            primary_category="spreadsheet",
            is_supported=True,
            suggested_route="table_pipeline",
            suggested_extractor="classify_table",
            classification={"type": "TABLE_FINANCIAL", "confidence": 0.99, "reason": "Has financial statements."}
        )

        mock_parse.return_value = "| Category | Q1 |\n|---|---|\n| Revenue | $50K |"
        mock_extract.return_value = {
            "image_type": "TABLE_FINANCIAL",
            "title": "Quarterly Financials",
            "reporting_period": "Q1 2026",
            "currency": "USD",
            "financial_metrics": [{"category": "Revenue", "values": {"Q1": "$50K"}}],
            "key_financial_takeaways": ["Takeaway"],
            "summary": "Balance sheet summary.",
            "rich_text_representation": "# Financial Report\nSummary"
        }

        response = self.client.post(
            "/process",
            files={
                "file": (
                    "balance_sheet.xlsx",
                    io.BytesIO(_FAKE_CSV),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            }
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "extract_financial_table")
        self.assertEqual(data["rich_text_representation"], "# Financial Report\nSummary")
        self.assertEqual(data["extracted_knowledge"]["title"], "Quarterly Financials")
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        mock_parse.assert_called_once()
        mock_extract.assert_called_once_with("| Category | Q1 |\n|---|---|\n| Revenue | $50K |")

    # ------------------------------------------------------------------
    # 7. Timeseries Spreadsheet Upload (CSV)
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.parse_table_file_to_markdown")
    @patch("RAG.routes.process.extract_timeseries_table")
    def test_07_process_timeseries_spreadsheet(self, mock_extract, mock_parse, mock_route):
        _sep("TEST 7 — Timeseries Spreadsheet routing")

        mock_route.return_value = RoutingMetadata(
            file_name="metrics.csv",
            file_extension="csv",
            mime_type="text/csv",
            primary_category="spreadsheet",
            is_supported=True,
            suggested_route="table_pipeline",
            suggested_extractor="classify_table",
            classification={"type": "TABLE_TIMESERIES", "confidence": 0.98, "reason": "Dates on index."}
        )

        mock_parse.return_value = "| Date | Count |\n|---|---|\n| 2026-01 | 100 |"
        mock_extract.return_value = {
            "image_type": "TABLE_TIMESERIES",
            "title": "Usage Over Time",
            "time_interval": "monthly",
            "timestamps": ["2026-01"],
            "series_data": [{"metric_name": "Usage", "values_over_time": ["100"]}],
            "trends_observed": ["Uptrend"],
            "summary": "Activity metric chart description.",
            "rich_text_representation": "# Chronological Trends\nUptrend"
        }

        response = self.client.post(
            "/process",
            files={"file": ("metrics.csv", io.BytesIO(_FAKE_CSV), "text/csv")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "extract_timeseries_table")
        self.assertEqual(data["rich_text_representation"], "# Chronological Trends\nUptrend")
        self.assertEqual(data["extracted_knowledge"]["time_interval"], "monthly")
        self.assertNotIn("rich_text_representation", data["extracted_knowledge"])
        mock_parse.assert_called_once()
        mock_extract.assert_called_once_with("| Date | Count |\n|---|---|\n| 2026-01 | 100 |")

    # ------------------------------------------------------------------
    # 8. PDF Orchestration Pipeline
    # ------------------------------------------------------------------
    @patch("RAG.routes.process.route_content")
    @patch("RAG.routes.process.orchestrate_pdf")
    def test_08_process_pdf_orchestration(self, mock_orchestrate, mock_route):
        _sep("TEST 8 — PDF Orchestration")

        mock_route.return_value = RoutingMetadata(
            file_name="report.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            primary_category="document",
            is_supported=True,
            suggested_route="pdf_pipeline",
            suggested_extractor="process_pdf",
            classification=None
        )

        from RAG.knowledge_normalizer import NormalizedKnowledgeChunk
        mock_orchestrate.return_value = [
            NormalizedKnowledgeChunk(
                chunk_id="fake-uuid-1",
                document_name="report.pdf",
                page_number=1,
                asset_type="text",
                classification_type="text",
                extractor_used="split_text",
                structured_knowledge={"text": "Hello PDF world."},
                rich_text_representation="Hello PDF world.",
                embedding_text="Hello PDF world.",
                metadata={"document_name": "report.pdf", "page_number": 1, "asset_type": "text", "classification_type": "text", "extractor_used": "split_text"}
            )
        ]

        response = self.client.post(
            "/process",
            files={"file": ("report.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["selected_extractor"], "document_orchestrator")
        self.assertEqual(data["rich_text_representation"], "Hello PDF world.")
        self.assertEqual(len(data["extracted_knowledge"]["chunks"]), 1)
        chunk = data["extracted_knowledge"]["chunks"][0]
        self.assertEqual(chunk["page_number"], 1)
        self.assertEqual(chunk["asset_type"], "text")
        self.assertEqual(chunk["classification_type"], "text")
        self.assertEqual(chunk["extractor_used"], "split_text")
        self.assertEqual(chunk["structured_knowledge"], {"text": "Hello PDF world."})
        mock_route.assert_called_once()
        mock_orchestrate.assert_called_once()

    # ------------------------------------------------------------------
    # 9. Unsupported Format
    # ------------------------------------------------------------------
    def test_09_process_unsupported_format(self):
        _sep("TEST 9 — Unsupported file format (.txt)")
        response = self.client.post(
            "/process",
            files={"file": ("notes.txt", io.BytesIO(_FAKE_TXT), "text/plain")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 415)
        self.assertIn("detail", data)
        self.assertIn("Unsupported file type", data["detail"])

    # ------------------------------------------------------------------
    # 10. Empty File Body
    # ------------------------------------------------------------------
    def test_10_process_empty_file(self):
        _sep("TEST 10 — Empty file body")
        response = self.client.post(
            "/process",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")}
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", data)
        self.assertIn("empty", data["detail"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
