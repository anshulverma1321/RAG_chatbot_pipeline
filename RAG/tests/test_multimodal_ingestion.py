"""
RAG/tests/test_multimodal_ingestion.py

Integration and unit tests for the updated multimodal ingestion flow.
Verifies orchestrator execution, SQLite database schema migration/extensions,
targeted Qdrant embeddings, metadata preservation, and backward compatibility.
"""

import os
import sys
import unittest
import tempfile
import sqlite3
import json
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from RAG.ingestion import process_pdf
from RAG.db import init_db, get_sibling_chunks
from RAG.vector_store import upsert_chunks
from RAG.knowledge_normalizer import normalize_knowledge_chunk


class TestMultimodalIngestion(unittest.TestCase):
    def setUp(self):
        # Create temp files for DB
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.vector_db_path = os.path.join(tempfile.gettempdir(), "qdrant_mock_dir")
        os.makedirs(self.vector_db_path, exist_ok=True)
        # Initialize db schema (this tests the CREATE TABLE logic)
        init_db(self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        try:
            os.remove(self.db_path)
        except Exception:
            pass
        import shutil
        try:
            shutil.rmtree(self.vector_db_path)
        except Exception:
            pass

    @patch("RAG.ingestion.get_file_hash", return_value="fake_hash_123")
    @patch("RAG.ingestion.PdfReader")
    @patch("RAG.ingestion.upsert_chunks")
    @patch("RAG.document_orchestrator.orchestrate_pdf")
    @patch("RAG.ingestion.os.path.exists", return_value=True)
    def test_01_multimodal_ingestion_flow(self, mock_exists, mock_orchestrate, mock_upsert, mock_pdf_reader, mock_get_hash):
        """Verify the full multimodal ingestion pipeline execution and SQLite/Qdrant mapping."""
        # 1. Setup mock PDF reader page count
        mock_reader_inst = MagicMock()
        mock_reader_inst.pages = [MagicMock()] * 2  # 2 pages
        mock_pdf_reader.return_value = mock_reader_inst

        # 2. Setup mock orchestrator outputs representing all required assets
        doc_name = "multimodal_test.pdf"
        chunks = [
            # 1. Standard text block
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=1,
                asset_type="text",
                classification_type="text",
                extractor_used="split_text",
                knowledge_object={"text": "Standard body text."},
                rich_text_representation="Standard body text."
            ),
            # 2. Chart image
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=1,
                asset_type="image",
                classification_type="CHART",
                extractor_used="extract_chart_knowledge",
                knowledge_object={"title": "Q1 Revenue Growth", "chart_type": "bar"},
                rich_text_representation="# Chart Details\nSome chart summary details."
            ),
            # 3. Diagram image
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=1,
                asset_type="image",
                classification_type="DIAGRAM",
                extractor_used="extract_diagram_knowledge",
                knowledge_object={"diagram_type": "flowchart", "nodes": ["A", "B"]},
                rich_text_representation="# Diagram Details\nSome diagram flow details."
            ),
            # 4. Infographic (mixed) image
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=2,
                asset_type="image",
                classification_type="MIXED",
                extractor_used="run_visual_understanding_logic",
                knowledge_object={"summary": "Mixed graphic content summary"},
                rich_text_representation="# Infographic Details\nMixed layout text."
            ),
            # 5. Natural image
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=2,
                asset_type="image",
                classification_type="NATURAL_IMAGE",
                extractor_used="describe_image_with_gemini",
                knowledge_object={"summary": "Scenic view of a mountain range"},
                rich_text_representation="# Scenic View Details\nNatural scenery summary."
            ),
            # 6. Financial table
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=2,
                asset_type="table",
                classification_type="TABLE_FINANCIAL",
                extractor_used="extract_financial_table",
                knowledge_object={"title": "Revenue Matrix", "currency": "EUR"},
                rich_text_representation="# Income Statement\nFinancial data summary."
            ),
            # 7. Statistical table
            normalize_knowledge_chunk(
                document_name=doc_name,
                page_number=2,
                asset_type="table",
                classification_type="TABLE_STATISTICAL",
                extractor_used="extract_statistical_table",
                knowledge_object={"title": "ML Model Benchmark", "variables": ["X", "Y"]},
                rich_text_representation="# Statistical Benchmark\nBenchmark results summary."
            )
        ]
        mock_orchestrate.return_value = chunks

        # 3. Run multimodal ingestion
        doc_id, msg = process_pdf("multimodal_test.pdf", self.db_path, self.vector_db_path)

        # 4. Verify orchestrator execution and return value
        self.assertEqual(doc_id, 1)
        self.assertIn("Successfully processed 'multimodal_test.pdf'", msg)
        mock_orchestrate.assert_called_once_with("multimodal_test.pdf")

        # 5. Verify SQLite persistence & metadata preservation
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number, sibling_order", (doc_id,)).fetchall()
        db_chunks = [dict(r) for r in rows]
        conn.close()

        self.assertEqual(len(db_chunks), 7)
        
        # Verify columns of first chunk (text)
        text_chunk = db_chunks[0]
        self.assertEqual(text_chunk["page_number"], 1)
        self.assertEqual(text_chunk["chunk_type"], "text")
        self.assertEqual(text_chunk["asset_type"], "text")
        self.assertEqual(text_chunk["classification_type"], "text")
        self.assertEqual(text_chunk["extractor_used"], "split_text")
        self.assertEqual(text_chunk["content"], "Standard body text.")
        self.assertEqual(text_chunk["document_name"], "multimodal_test.pdf")
        self.assertIsNotNone(text_chunk["metadata"])
        
        # Verify chart metadata
        chart_chunk = db_chunks[1]
        self.assertEqual(chart_chunk["chunk_type"], "image")
        self.assertEqual(chart_chunk["asset_type"], "image")
        self.assertEqual(chart_chunk["classification_type"], "CHART")
        self.assertEqual(chart_chunk["extractor_used"], "extract_chart_knowledge")
        self.assertEqual(chart_chunk["content"], "# Chart Details\nSome chart summary details.")
        
        # Verify financial table metadata
        fin_chunk = db_chunks[5]
        self.assertEqual(fin_chunk["chunk_type"], "table")
        self.assertEqual(fin_chunk["asset_type"], "table")
        self.assertEqual(fin_chunk["classification_type"], "TABLE_FINANCIAL")
        self.assertEqual(fin_chunk["extractor_used"], "extract_financial_table")
        self.assertEqual(fin_chunk["content"], "# Income Statement\nFinancial data summary.")
        meta_dict = json.loads(fin_chunk["metadata"])
        self.assertEqual(meta_dict["classification_type"], "TABLE_FINANCIAL")

        # 6. Verify Qdrant persistence mapping
        mock_upsert.assert_called_once()
        qdrant_args = mock_upsert.call_args[0][1]
        self.assertEqual(len(qdrant_args), 7)
        
        # Verify first upsert payload dict has correct metadata keys and embedding source
        first_qdrant = qdrant_args[0]
        self.assertEqual(first_qdrant["document_id"], 1)
        self.assertEqual(first_qdrant["page_number"], 1)
        self.assertEqual(first_qdrant["asset_type"], "text")
        self.assertEqual(first_qdrant["classification_type"], "text")
        self.assertEqual(first_qdrant["extractor_used"], "split_text")
        self.assertEqual(first_qdrant["rich_text_representation"], "Standard body text.")
        self.assertEqual(first_qdrant["embedding_text"], "Standard body text.")

    def test_02_sqlite_schema_migration(self):
        """Test that init_db correctly adds missing columns to an existing SQLite chunks table."""
        # 1. Create a mock connection to a database that has the chunks table in its old format
        temp_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        try:
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            # Create old chunks table without the new metadata columns
            cursor.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                document_id INTEGER,
                page_number INTEGER,
                chunk_type TEXT,
                content TEXT NOT NULL,
                sibling_order INTEGER
            )
            """)
            conn.commit()
            conn.close()

            # 2. Run init_db on the database. It should run the migration checks and run ALTER TABLE.
            init_db(temp_db_path)

            # 3. Check table columns
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chunks)")
            cols = [row[1] for row in cursor.fetchall()]
            conn.close()

            # Verify that the new columns now exist
            for expected_col in ["asset_type", "classification_type", "extractor_used", "document_name", "metadata"]:
                self.assertIn(expected_col, cols)
        finally:
            os.close(temp_fd)
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

    @patch("RAG.vector_store.get_vector_store")
    def test_03_upsert_chunks_embedding_source_and_compatibility(self, mock_get_store):
        """Test targeted embeddings on embedding_text and backward compatibility of Qdrant payloads."""
        mock_store_inst = MagicMock()
        mock_get_store.return_value = mock_store_inst

        # A. Mock Legacy payload dictionary structure
        legacy_chunks = [
            {
                "id": "legacy-uuid-1",
                "document_id": 10,
                "page_number": 3,
                "chunk_type": "text",
                "content": "Legacy textual content",
                "filename": "legacy.pdf"
            }
        ]

        # B. Mock Normalized chunk dictionary structure (Phase 6)
        normalized_chunks = [
            {
                "chunk_id": "99999999-9999-9999-9999-999999999999",
                "document_id": 11,
                "page_number": 4,
                "asset_type": "image",
                "classification_type": "CHART",
                "extractor_used": "extract_chart_knowledge",
                "document_name": "multimodal_test.pdf",
                "structured_knowledge": {"insights": ["sales are up"]},
                "rich_text_representation": "# Chart Summary\nSales up",
                "embedding_text": "Chart Title: Profit\nSummary:\nSales up",
                "metadata": {"custom_tag": "test"}
            }
        ]

        # 1. Run legacy upsert - verify fallback works
        upsert_chunks(self.vector_db_path, legacy_chunks)
        self.assertTrue(mock_store_inst.add_documents.called)
        
        legacy_doc = mock_store_inst.add_documents.call_args_list[0][1]["documents"][0]
        self.assertEqual(legacy_doc.page_content, "Legacy textual content")
        self.assertEqual(legacy_doc.metadata["document_id"], 10)
        self.assertEqual(legacy_doc.metadata["chunk_type"], "text")
        self.assertEqual(legacy_doc.metadata["filename"], "legacy.pdf")

        # Reset call log
        mock_store_inst.reset_mock()

        # 2. Run normalized upsert - verify embedding source targeting and payload schema
        upsert_chunks(self.vector_db_path, normalized_chunks)
        self.assertTrue(mock_store_inst.add_documents.called)
        
        norm_doc = mock_store_inst.add_documents.call_args_list[0][1]["documents"][0]
        # The content to embed MUST strictly be chunk.embedding_text
        self.assertEqual(norm_doc.page_content, "Chart Title: Profit\nSummary:\nSales up")
        # Metadata values must preserve both legacy keys and new spec keys
        self.assertEqual(norm_doc.metadata["document_id"], 11)
        self.assertEqual(norm_doc.metadata["page_number"], 4)
        self.assertEqual(norm_doc.metadata["chunk_type"], "image")
        self.assertEqual(norm_doc.metadata["filename"], "multimodal_test.pdf")
        self.assertEqual(norm_doc.metadata["chunk_id"], "99999999-9999-9999-9999-999999999999")
        self.assertEqual(norm_doc.metadata["document_name"], "multimodal_test.pdf")
        self.assertEqual(norm_doc.metadata["asset_type"], "image")
        self.assertEqual(norm_doc.metadata["classification_type"], "CHART")
        self.assertEqual(norm_doc.metadata["extractor_used"], "extract_chart_knowledge")
        self.assertEqual(norm_doc.metadata["rich_text_representation"], "# Chart Summary\nSales up")
        self.assertEqual(norm_doc.metadata["metadata"], {"custom_tag": "test"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
