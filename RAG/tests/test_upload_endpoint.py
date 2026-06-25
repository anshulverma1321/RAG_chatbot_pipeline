"""
RAG/tests/test_upload_endpoint.py

Phase 8 — Universal Upload API Test Suite (Updated)
=====================================================
Tests POST /upload endpoint behaviour for all supported and unsupported file types.

Phase 8 changes:
  - POST /upload now performs FULL ingestion (not routing-only).
  - Response schema is UploadIngestionResponse, not UploadResponse.
  - All DB/vector-store/orchestrator calls are mocked.
  - Error cases (415, 400) are unchanged.
"""

import io
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure the project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app
from RAG.knowledge_normalizer import NormalizedKnowledgeChunk

# ---------------------------------------------------------------------------
# Minimal synthetic file payloads
# ---------------------------------------------------------------------------
_FAKE_PDF   = b"%PDF-1.4 fake pdf bytes"
_FAKE_PNG   = b"\x89PNG\r\n\x1a\n fake png bytes"
_FAKE_JPG   = b"\xff\xd8\xff fake jpg bytes"
_FAKE_JPEG  = b"\xff\xd8\xff fake jpeg bytes"
_FAKE_CSV   = b"Name,Age,Score\nAlice,30,95\nBob,25,88\n"
_FAKE_XLSX  = b"PK fake xlsx bytes"
_FAKE_TXT   = b"This is a plain text file."


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _text_chunk(name: str, page: int = 1) -> NormalizedKnowledgeChunk:
    return NormalizedKnowledgeChunk(
        chunk_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        document_name=name, page_number=page,
        asset_type="text", classification_type="text", extractor_used="split_text",
        structured_knowledge={"text": "Sample text."},
        rich_text_representation="Sample text.",
        embedding_text="Sample text.",
        metadata={"document_name": name, "page_number": page,
                  "asset_type": "text", "classification_type": "text",
                  "extractor_used": "split_text"},
    )


def _image_chunk(name: str) -> NormalizedKnowledgeChunk:
    return NormalizedKnowledgeChunk(
        chunk_id="11111111-2222-3333-4444-555555555555",
        document_name=name, page_number=1,
        asset_type="image", classification_type="CHART", extractor_used="extract_chart_knowledge",
        structured_knowledge={"title": "Chart"},
        rich_text_representation="# Chart",
        embedding_text="Chart Title: Chart",
        metadata={"document_name": name, "page_number": 1,
                  "asset_type": "image", "classification_type": "CHART",
                  "extractor_used": "extract_chart_knowledge"},
    )


def _table_chunk(name: str) -> NormalizedKnowledgeChunk:
    return NormalizedKnowledgeChunk(
        chunk_id="ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb",
        document_name=name, page_number=1,
        asset_type="table", classification_type="TABLE_SIMPLE", extractor_used="extract_simple_table",
        structured_knowledge={"title": "Table"},
        rich_text_representation="# Table",
        embedding_text="Table Title: Table",
        metadata={"document_name": name, "page_number": 1,
                  "asset_type": "table", "classification_type": "TABLE_SIMPLE",
                  "extractor_used": "extract_simple_table"},
    )


# ---------------------------------------------------------------------------
# Shared mock patch stack for ingestion tests
# ---------------------------------------------------------------------------
_INGEST_PATCHES = [
    patch("RAG.db.get_document_by_hash", return_value=None),
    patch("RAG.db.add_document", return_value=1),
    patch("RAG.routes.upload._store_chunks"),
]


class TestUniversalUploadEndpoint(unittest.TestCase):
    """Phase 8 integration tests for POST /upload — full ingestion pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    # ------------------------------------------------------------------
    # 1. PDF upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=1)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload.orchestrate_pdf")
    @patch("RAG.routes.upload.route_content")
    def test_01_pdf_upload(self, mock_route, mock_orchestrate, mock_store, *_):
        _sep("TEST 1 — PDF upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="sample_report.pdf", file_extension="pdf",
            mime_type="application/pdf", primary_category="document",
            is_supported=True, suggested_route="pdf_pipeline",
            suggested_extractor="process_pdf", classification=None,
        )
        mock_orchestrate.return_value = [
            _text_chunk("sample_report.pdf", 1),
            _text_chunk("sample_report.pdf", 2),
        ]

        response = self.client.post(
            "/upload",
            files={"file": ("sample_report.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_name"], "sample_report.pdf")
        self.assertEqual(data["file_type"], "pdf")
        self.assertTrue(data["ingested"])
        self.assertEqual(data["text_chunks"], 2)
        self.assertEqual(data["table_chunks"], 0)
        self.assertEqual(data["image_chunks"], 0)
        self.assertEqual(data["total_chunks"], 2)
        self.assertIsInstance(data["document_id"], int)
        self.assertGreaterEqual(data["pages_processed"], 1)
        mock_store.assert_called_once()

    # ------------------------------------------------------------------
    # 2. PNG image upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=2)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_image_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_02_png_image_upload(self, mock_route, mock_img, mock_store, *_):
        _sep("TEST 2 — PNG image upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="chart.png", file_extension="png",
            mime_type="image/png", primary_category="image",
            is_supported=True, suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "CHART", "confidence": 0.95, "reason": "Bar chart"},
        )
        mock_img.return_value = [_image_chunk("chart.png")]

        response = self.client.post(
            "/upload",
            files={"file": ("chart.png", io.BytesIO(_FAKE_PNG), "image/png")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_name"], "chart.png")
        self.assertEqual(data["file_type"], "png")
        self.assertTrue(data["ingested"])
        self.assertEqual(data["image_chunks"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["pages_processed"], 1)
        mock_img.assert_called_once()

    # ------------------------------------------------------------------
    # 3. JPG image upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=3)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_image_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_03_jpg_image_upload(self, mock_route, mock_img, mock_store, *_):
        _sep("TEST 3 — JPG image upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="diagram.jpg", file_extension="jpg",
            mime_type="image/jpeg", primary_category="image",
            is_supported=True, suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "DIAGRAM", "confidence": 0.90, "reason": "Flowchart"},
        )
        mock_img.return_value = [_image_chunk("diagram.jpg")]

        response = self.client.post(
            "/upload",
            files={"file": ("diagram.jpg", io.BytesIO(_FAKE_JPG), "image/jpeg")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_type"], "jpg")
        self.assertTrue(data["ingested"])

    # ------------------------------------------------------------------
    # 4. CSV spreadsheet upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=4)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_spreadsheet_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_04_csv_upload(self, mock_route, mock_tbl, mock_store, *_):
        _sep("TEST 4 — CSV spreadsheet upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="financials.csv", file_extension="csv",
            mime_type="text/csv", primary_category="spreadsheet",
            is_supported=True, suggested_route="table_pipeline",
            suggested_extractor="classify_table",
            classification={"type": "TABLE_SIMPLE", "confidence": 0.88, "reason": "Simple data"},
        )
        mock_tbl.return_value = [_table_chunk("financials.csv")]

        response = self.client.post(
            "/upload",
            files={"file": ("financials.csv", io.BytesIO(_FAKE_CSV), "text/csv")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_name"], "financials.csv")
        self.assertEqual(data["file_type"], "csv")
        self.assertTrue(data["ingested"])
        self.assertEqual(data["table_chunks"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["pages_processed"], 1)
        mock_tbl.assert_called_once()

    # ------------------------------------------------------------------
    # 5. XLSX upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=5)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_spreadsheet_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_05_xlsx_upload(self, mock_route, mock_tbl, mock_store, *_):
        _sep("TEST 5 — XLSX spreadsheet upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="revenue.xlsx", file_extension="xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            primary_category="spreadsheet", is_supported=True,
            suggested_route="table_pipeline", suggested_extractor="classify_table",
            classification={"type": "TABLE_FINANCIAL", "confidence": 0.97, "reason": "Financial data"},
        )
        mock_tbl.return_value = [_table_chunk("revenue.xlsx")]

        response = self.client.post(
            "/upload",
            files={"file": (
                "revenue.xlsx",
                io.BytesIO(_FAKE_XLSX),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_name"], "revenue.xlsx")
        self.assertEqual(data["file_type"], "xlsx")
        self.assertTrue(data["ingested"])

    # ------------------------------------------------------------------
    # 6. Unsupported file type → HTTP 415
    # ------------------------------------------------------------------
    def test_06_unsupported_file_type(self):
        _sep("TEST 6 — Unsupported file type (.txt)")
        response = self.client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(_FAKE_TXT), "text/plain")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 415)
        self.assertIn("detail", data)
        self.assertIn("Unsupported file type", data["detail"])

    # ------------------------------------------------------------------
    # 7. Empty file body → HTTP 400
    # ------------------------------------------------------------------
    def test_07_empty_file(self):
        _sep("TEST 7 — Empty file upload")
        response = self.client.post(
            "/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", data)
        self.assertIn("empty", data["detail"].lower())

    # ------------------------------------------------------------------
    # 8. Verify UploadIngestionResponse schema completeness
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=8)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload.orchestrate_pdf")
    @patch("RAG.routes.upload.route_content")
    def test_08_ingestion_response_schema(self, mock_route, mock_orchestrate, mock_store, *_):
        _sep("TEST 8 — UploadIngestionResponse schema validation")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="document.pdf", file_extension="pdf",
            mime_type="application/pdf", primary_category="document",
            is_supported=True, suggested_route="pdf_pipeline",
            suggested_extractor="process_pdf", classification=None,
        )
        mock_orchestrate.return_value = [_text_chunk("document.pdf", 1)]

        response = self.client.post(
            "/upload",
            files={"file": ("document.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        required_fields = {
            "status", "document_id", "file_name", "file_type",
            "pages_processed", "text_chunks", "table_chunks",
            "image_chunks", "total_chunks", "ingested",
        }
        for field in required_fields:
            self.assertIn(field, data, msg=f"Missing field in response: {field}")

        print("All required UploadIngestionResponse fields present:", sorted(required_fields))
        print("✓ Schema validated successfully.")

    # ------------------------------------------------------------------
    # 9. JPEG upload → success response
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=9)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_image_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_09_jpeg_upload(self, mock_route, mock_img, mock_store, *_):
        _sep("TEST 9 — JPEG image upload")
        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="photo.jpeg", file_extension="jpeg",
            mime_type="image/jpeg", primary_category="image",
            is_supported=True, suggested_route="image_pipeline",
            suggested_extractor="classify_image",
            classification={"type": "NATURAL_IMAGE", "confidence": 0.93, "reason": "Photograph"},
        )
        mock_img.return_value = [_image_chunk("photo.jpeg")]

        response = self.client.post(
            "/upload",
            files={"file": ("photo.jpeg", io.BytesIO(_FAKE_JPEG), "image/jpeg")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_type"], "jpeg")
        self.assertTrue(data["ingested"])
        self.assertEqual(data["image_chunks"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
