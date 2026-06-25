"""
RAG/tests/test_public_api_workflow.py

Phase 8 — Public API Workflow Test Suite
=========================================
Validates the production-ready single-endpoint architecture:

  POST /upload  — full ingestion pipeline (the only public ingestion endpoint)
  POST /query   — unchanged retrieval endpoint

Tests:
  1. Upload PDF   → ingest success   → UploadIngestionResponse schema
  2. Upload image → ingest success   → UploadIngestionResponse schema
  3. Upload spreadsheet → ingest success → UploadIngestionResponse schema
  4. Duplicate upload   → HTTP 200 already_ingested (not an error)
  5. POST /process hidden from Swagger /openapi.json
  6. POST /ingest  hidden from Swagger /openapi.json
  7. POST /query   works after upload (mock query engine)

All external I/O (Gemini API, SQLite, Qdrant, orchestrate_pdf, extractors) is mocked
so tests are fast, deterministic, and require no real credentials or files.
"""

import io
import os
import sys
import json
import hashlib
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app
from RAG.knowledge_normalizer import NormalizedKnowledgeChunk

# ---------------------------------------------------------------------------
# Minimal synthetic payloads
# ---------------------------------------------------------------------------
_FAKE_PDF = b"%PDF-1.4 fake pdf bytes for phase 8 test"
_FAKE_PNG = b"\x89PNG\r\n\x1a\n fake png content"
_FAKE_JPG = b"\xff\xd8\xff fake jpg content"
_FAKE_JPEG = b"\xff\xd8\xff fake jpeg content"
_FAKE_CSV = b"Product,Revenue,Quarter\nWidgetA,50000,Q1\nWidgetB,75000,Q2\n"
_FAKE_XLSX = b"PK fake xlsx bytes"
_FAKE_TXT = b"This is plain text - unsupported"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sep(title: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print("=" * 66)


def _make_text_chunk(name: str = "test.pdf", page: int = 1) -> NormalizedKnowledgeChunk:
    """Creates a minimal NormalizedKnowledgeChunk of type 'text'."""
    return NormalizedKnowledgeChunk(
        chunk_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        document_name=name,
        page_number=page,
        asset_type="text",
        classification_type="text",
        extractor_used="split_text",
        structured_knowledge={"text": "Sample extracted text content."},
        rich_text_representation="Sample extracted text content.",
        embedding_text="Sample extracted text content.",
        metadata={
            "document_name": name,
            "page_number": page,
            "asset_type": "text",
            "classification_type": "text",
            "extractor_used": "split_text",
        },
    )


def _make_image_chunk(name: str = "chart.png") -> NormalizedKnowledgeChunk:
    """Creates a minimal NormalizedKnowledgeChunk of type 'image'."""
    return NormalizedKnowledgeChunk(
        chunk_id="11111111-2222-3333-4444-555555555555",
        document_name=name,
        page_number=1,
        asset_type="image",
        classification_type="CHART",
        extractor_used="extract_chart_knowledge",
        structured_knowledge={"chart_type": "bar_chart", "title": "Revenue Chart"},
        rich_text_representation="# Revenue Chart\nBar chart showing quarterly revenue.",
        embedding_text="Chart Title: Revenue Chart\nChart Type: bar_chart",
        metadata={
            "document_name": name,
            "page_number": 1,
            "asset_type": "image",
            "classification_type": "CHART",
            "extractor_used": "extract_chart_knowledge",
        },
    )


def _make_table_chunk(name: str = "financials.csv") -> NormalizedKnowledgeChunk:
    """Creates a minimal NormalizedKnowledgeChunk of type 'table'."""
    return NormalizedKnowledgeChunk(
        chunk_id="ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb",
        document_name=name,
        page_number=1,
        asset_type="table",
        classification_type="TABLE_FINANCIAL",
        extractor_used="extract_financial_table",
        structured_knowledge={"title": "Quarterly Financials", "summary": "Q1/Q2 revenue data."},
        rich_text_representation="# Quarterly Financials\nQ1: $50K | Q2: $75K",
        embedding_text="Table Title: Quarterly Financials\nTable Summary: Q1/Q2 revenue data.",
        metadata={
            "document_name": name,
            "page_number": 1,
            "asset_type": "table",
            "classification_type": "TABLE_FINANCIAL",
            "extractor_used": "extract_financial_table",
        },
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestPublicAPIWorkflow(unittest.TestCase):
    """Phase 8 — Public API Workflow Integration Tests."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    # ------------------------------------------------------------------
    # Test 1: Upload PDF → full ingestion → UploadIngestionResponse
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=42)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload.orchestrate_pdf")
    @patch("RAG.routes.upload.route_content")
    def test_01_upload_pdf_ingest_success(
        self,
        mock_route,
        mock_orchestrate,
        mock_store,
        mock_add_doc,
        mock_get_hash,
    ):
        _sep("TEST 1 — Upload PDF → ingest success")

        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="report.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            primary_category="document",
            is_supported=True,
            suggested_route="pdf_pipeline",
            suggested_extractor="process_pdf",
            classification=None,
        )

        mock_orchestrate.return_value = [
            _make_text_chunk("report.pdf", 1),
            _make_text_chunk("report.pdf", 2),
            _make_text_chunk("report.pdf", 3),
        ]

        response = self.client.post(
            "/upload",
            files={"file": ("report.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["file_name"], "report.pdf")
        self.assertEqual(data["file_type"], "pdf")
        self.assertIsInstance(data["document_id"], int)
        self.assertGreaterEqual(data["pages_processed"], 1)
        self.assertEqual(data["text_chunks"], 3)
        self.assertEqual(data["table_chunks"], 0)
        self.assertEqual(data["image_chunks"], 0)
        self.assertEqual(data["total_chunks"], 3)
        self.assertTrue(data["ingested"])
        self.assertIsNone(data.get("message"))

        mock_orchestrate.assert_called_once()
        mock_store.assert_called_once()
        print("✓ PDF ingestion response schema validated.")

    # ------------------------------------------------------------------
    # Test 2: Upload image → full ingestion → UploadIngestionResponse
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=7)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_image_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_02_upload_image_ingest_success(
        self,
        mock_route,
        mock_image_chunks,
        mock_store,
        mock_add_doc,
        mock_get_hash,
    ):
        _sep("TEST 2 — Upload image → ingest success")

        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="chart.png",
            file_extension="png",
            mime_type="image/png",
            primary_category="image",
            is_supported=True,
            suggested_route="image_pipeline",
            suggested_extractor="extract_chart_knowledge",
            classification={"type": "CHART", "confidence": 0.97, "reason": "Bar chart."},
        )
        mock_image_chunks.return_value = [_make_image_chunk("chart.png")]

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
        self.assertEqual(data["document_id"], 7)
        self.assertEqual(data["pages_processed"], 1)
        self.assertEqual(data["text_chunks"], 0)
        self.assertEqual(data["table_chunks"], 0)
        self.assertEqual(data["image_chunks"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertTrue(data["ingested"])

        mock_image_chunks.assert_called_once()
        mock_store.assert_called_once()
        print("✓ Image ingestion response schema validated.")

    # ------------------------------------------------------------------
    # Test 3: Upload spreadsheet → full ingestion → UploadIngestionResponse
    # ------------------------------------------------------------------
    @patch("RAG.routes.upload.get_document_by_hash", return_value=None)
    @patch("RAG.routes.upload.add_document", return_value=15)
    @patch("RAG.routes.upload._store_chunks")
    @patch("RAG.routes.upload._ingest_spreadsheet_to_chunks")
    @patch("RAG.routes.upload.route_content")
    def test_03_upload_spreadsheet_ingest_success(
        self,
        mock_route,
        mock_table_chunks,
        mock_store,
        mock_add_doc,
        mock_get_hash,
    ):
        _sep("TEST 3 — Upload spreadsheet → ingest success")

        from RAG.content_router import RoutingMetadata
        mock_route.return_value = RoutingMetadata(
            file_name="financials.csv",
            file_extension="csv",
            mime_type="text/csv",
            primary_category="spreadsheet",
            is_supported=True,
            suggested_route="table_pipeline",
            suggested_extractor="extract_financial_table",
            classification={"type": "TABLE_FINANCIAL", "confidence": 0.99, "reason": "Financial data."},
        )
        mock_table_chunks.return_value = [_make_table_chunk("financials.csv")]

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
        self.assertEqual(data["document_id"], 15)
        self.assertEqual(data["pages_processed"], 1)
        self.assertEqual(data["text_chunks"], 0)
        self.assertEqual(data["table_chunks"], 1)
        self.assertEqual(data["image_chunks"], 0)
        self.assertEqual(data["total_chunks"], 1)
        self.assertTrue(data["ingested"])

        mock_table_chunks.assert_called_once()
        mock_store.assert_called_once()
        print("✓ Spreadsheet ingestion response schema validated.")

    # ------------------------------------------------------------------
    # Test 4: Duplicate upload → HTTP 200 + already_ingested (not an error)
    # ------------------------------------------------------------------
    @patch(
        "RAG.routes.upload.get_document_by_hash",
        return_value={"id": 99, "filename": "report.pdf", "file_hash": _sha256(_FAKE_PDF)},
    )
    def test_04_duplicate_upload_returns_already_ingested(self, mock_get_hash):
        _sep("TEST 4 — Duplicate upload → HTTP 200 already_ingested")

        response = self.client.post(
            "/upload",
            files={"file": ("report.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        # Must be HTTP 200, NOT 500
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "already_ingested")
        self.assertEqual(data["document_id"], 99)
        self.assertEqual(data["file_name"], "report.pdf")
        self.assertFalse(data["ingested"])
        self.assertEqual(data["total_chunks"], 0)
        self.assertIn("already exists", data["message"])
        print("✓ Duplicate returns HTTP 200 with already_ingested status.")

    # ------------------------------------------------------------------
    # Test 5: POST /process is hidden from Swagger /openapi.json
    # ------------------------------------------------------------------
    def test_05_process_endpoint_hidden_from_swagger(self):
        _sep("TEST 5 — POST /process hidden from Swagger")

        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        openapi = response.json()

        paths = openapi.get("paths", {})
        print("Public paths in Swagger:", list(paths.keys()))

        self.assertNotIn(
            "/process",
            paths,
            msg="POST /process should NOT appear in Swagger (include_in_schema=False)",
        )
        print("✓ /process is NOT in Swagger paths.")

    # ------------------------------------------------------------------
    # Test 6: POST /ingest is hidden from Swagger /openapi.json
    # ------------------------------------------------------------------
    def test_06_ingest_endpoint_hidden_from_swagger(self):
        _sep("TEST 6 — POST /ingest hidden from Swagger")

        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        openapi = response.json()

        paths = openapi.get("paths", {})
        print("Public paths in Swagger:", list(paths.keys()))

        self.assertNotIn(
            "/ingest",
            paths,
            msg="POST /ingest should NOT appear in Swagger (include_in_schema=False)",
        )
        print("✓ /ingest is NOT in Swagger paths.")

    # ------------------------------------------------------------------
    # Test 7: POST /query works (mock query engine) — unchanged workflow
    # ------------------------------------------------------------------
    @patch("RAG.app.execute_rag_query", return_value="The quarterly revenue was $50K in Q1.")
    def test_07_query_works_after_upload(self, mock_query):
        _sep("TEST 7 — POST /query works after upload")

        response = self.client.post(
            "/query",
            json={"query": "What was the quarterly revenue?"},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", data)
        self.assertIn("revenue", data["answer"].lower())

        mock_query.assert_called_once_with(
            query="What was the quarterly revenue?",
            db_path=mock_query.call_args.kwargs.get("db_path") or mock_query.call_args[1].get("db_path") or mock_query.call_args[0][1],
            vector_db_path=mock_query.call_args.kwargs.get("vector_db_path") or mock_query.call_args[1].get("vector_db_path") or mock_query.call_args[0][2],
            document_ids=None,
            top_k=5,
        )
        print("✓ Query endpoint returns answer.")

    # ------------------------------------------------------------------
    # Test 8: Verify public Swagger paths — only /upload and /query visible
    # ------------------------------------------------------------------
    def test_08_only_upload_and_query_in_swagger(self):
        _sep("TEST 8 — Only /upload and /query visible in Swagger")

        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        openapi = response.json()

        paths = set(openapi.get("paths", {}).keys())
        print("All Swagger paths:", sorted(paths))

        # Must be present
        self.assertIn("/upload", paths, msg="/upload must be visible in Swagger")
        self.assertIn("/query", paths, msg="/query must be visible in Swagger")

        # Must be hidden
        self.assertNotIn("/process", paths, msg="/process must be hidden from Swagger")
        self.assertNotIn("/ingest", paths, msg="/ingest must be hidden from Swagger")

        print("✓ Public surface: /upload, /query — Internal hidden: /process, /ingest")

    # ------------------------------------------------------------------
    # Test 9: Unsupported file type still returns HTTP 415
    # ------------------------------------------------------------------
    def test_09_unsupported_file_type_returns_415(self):
        _sep("TEST 9 — Unsupported file type (.txt) → HTTP 415")

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
        print("✓ Unsupported type correctly rejected with HTTP 415.")

    # ------------------------------------------------------------------
    # Test 10: Empty file body still returns HTTP 400
    # ------------------------------------------------------------------
    def test_10_empty_file_returns_400(self):
        _sep("TEST 10 — Empty file body → HTTP 400")

        response = self.client.post(
            "/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        print("Status code:", response.status_code)
        data = response.json()
        print("Response:\n", json.dumps(data, indent=2))

        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", data["detail"].lower())
        print("✓ Empty file correctly rejected with HTTP 400.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
