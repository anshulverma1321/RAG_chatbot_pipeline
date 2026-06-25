"""
RAG/tests/test_multimodal_retrieval.py
Unit tests for multimodal asset-aware retrieval, grouping, and context assembly.
"""

import os
import sys
import unittest
import tempfile
import sqlite3
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from RAG.query_engine import execute_rag_query, RetrievalAsset
from RAG.db import init_db, add_chunks, add_document


class TestMultimodalRetrieval(unittest.TestCase):
    def setUp(self):
        # Create temp files for DB
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.vector_db_path = os.path.join(tempfile.gettempdir(), "qdrant_mock_dir")
        os.makedirs(self.vector_db_path, exist_ok=True)
        # Initialize db schema
        init_db(self.db_path)
        
        # Ingest a mock document to get a document ID
        self.doc_id = add_document(self.db_path, "multimodal_report.pdf", "hash123", 5)
        
        # Define and insert mock chunks representing multiple asset types with structured knowledge
        self.mock_chunks = [
            # 1. Text chunk
            {
                "id": "chunk-text-1",
                "document_id": self.doc_id,
                "page_number": 1,
                "chunk_type": "text",
                "content": "Standard paragraph introducing sensors.",
                "sibling_order": 0,
                "asset_type": "text",
                "classification_type": "text",
                "extractor_used": "split_text",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {"text": "Standard paragraph introducing sensors."}
                }
            },
            # 2. Text chunk 2 (on page 1)
            {
                "id": "chunk-text-2",
                "document_id": self.doc_id,
                "page_number": 1,
                "chunk_type": "text",
                "content": "Another textual paragraph about hardware.",
                "sibling_order": 1,
                "asset_type": "text",
                "classification_type": "text",
                "extractor_used": "split_text",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {"text": "Another textual paragraph about hardware."}
                }
            },
            # 3. Chart image
            {
                "id": "chunk-chart-1",
                "document_id": self.doc_id,
                "page_number": 2,
                "chunk_type": "image",
                "content": "# Revenue growth chart summary",
                "sibling_order": 0,
                "asset_type": "image",
                "classification_type": "CHART",
                "extractor_used": "extract_chart_knowledge",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {
                        "insights": ["Highest spike in Q2.", "Consistent yearly uptrend."],
                        "title": "Quarterly Revenue"
                    }
                }
            },
            # 4. Diagram image
            {
                "id": "chunk-diagram-1",
                "document_id": self.doc_id,
                "page_number": 3,
                "chunk_type": "image",
                "content": "# Flow diagram summary",
                "sibling_order": 0,
                "asset_type": "image",
                "classification_type": "DIAGRAM",
                "extractor_used": "extract_diagram_knowledge",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {
                        "relationships": [{"from": "Client", "to": "Server", "label_or_relationship": "HTTPS"}],
                        "workflow": ["Initiate hand-shake", "Authorize client token"]
                    }
                }
            },
            # 5. Financial table
            {
                "id": "chunk-table-1",
                "document_id": self.doc_id,
                "page_number": 4,
                "chunk_type": "table",
                "content": "# Q1 Balance Sheet summary",
                "sibling_order": 0,
                "asset_type": "table",
                "classification_type": "TABLE_FINANCIAL",
                "extractor_used": "extract_financial_table",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {
                        "key_financial_takeaways": ["OpEx is down 5%."],
                        "financial_metrics": [{"category": "Revenue", "values": {"Q1": "$100K"}}]
                    }
                }
            },
            # 6. Statistical table
            {
                "id": "chunk-table-2",
                "document_id": self.doc_id,
                "page_number": 5,
                "chunk_type": "table",
                "content": "# Model Benchmark summary",
                "sibling_order": 0,
                "asset_type": "table",
                "classification_type": "TABLE_STATISTICAL",
                "extractor_used": "extract_statistical_table",
                "document_name": "multimodal_report.pdf",
                "metadata": {
                    "structured_knowledge": {
                        "statistical_conclusions": ["Treatment is statistically significant."],
                        "data_summary": [{"variable_or_group": "Treatment", "metric_values": {"p-val": "<0.005"}}]
                    }
                }
            }
        ]
        add_chunks(self.db_path, self.mock_chunks)

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

    @patch("RAG.query_engine.get_query_embedding", return_value=[0.1]*3072)
    @patch("RAG.query_engine.search_vectors")
    @patch("RAG.query_engine.ChatGoogleGenerativeAI")
    def test_01_retrieval_grouping_and_text_suppression_prevention(self, mock_chat, mock_search, mock_embed):
        """Verify that grouping prevents multiple text chunks from suppressing chart and table assets."""
        # 1. Setup mock search hits:
        # returns 2 hits from page 1 (text), 1 hit from page 2 (chart), 1 hit from page 4 (financial table)
        mock_hits = [
            {"id": "chunk-text-1", "score": 0.90, "payload": {"document_id": self.doc_id, "page_number": 1, "chunk_type": "text", "filename": "multimodal_report.pdf"}},
            {"id": "chunk-text-2", "score": 0.89, "payload": {"document_id": self.doc_id, "page_number": 1, "chunk_type": "text", "filename": "multimodal_report.pdf"}},
            {"id": "chunk-chart-1", "score": 0.85, "payload": {"document_id": self.doc_id, "page_number": 2, "chunk_type": "image", "filename": "multimodal_report.pdf"}},
            {"id": "chunk-table-1", "score": 0.80, "payload": {"document_id": self.doc_id, "page_number": 4, "chunk_type": "table", "filename": "multimodal_report.pdf"}}
        ]
        mock_search.return_value = mock_hits

        # Setup mock Chat invoking output
        mock_chat_inst = MagicMock()
        mock_chat_inst.invoke.return_value = AIMessage(content="Grounded Answer.")
        mock_chat_inst.return_value = AIMessage(content="Grounded Answer.")
        mock_chat.return_value = mock_chat_inst

        # 2. Run retrieval query with top_k=3
        execute_rag_query("Compare quarterly opex", self.db_path, self.vector_db_path, top_k=3)

        # 3. Assert on prompt contents passed to invoke
        call_args = mock_chat_inst.invoke.call_args or mock_chat_inst.call_args
        prompt_tmpl_arg = call_args[0][0]
        messages = prompt_tmpl_arg.messages
        user_prompt = messages[1].content if hasattr(messages[1], "content") else str(messages[1])

        # Verify grouping results:
        # Grouped keys should be:
        # - (multimodal_report.pdf, 1, text) -> best score 0.90
        # - (multimodal_report.pdf, 2, image) -> best score 0.85
        # - (multimodal_report.pdf, 4, table) -> best score 0.80
        # Since top_k=3, all three unique groups must be retrieved!
        # Check that context_str has all 3 sources
        self.assertIn("Page 1, Type: text", user_prompt)
        self.assertIn("Page 2, Type: image", user_prompt)
        self.assertIn("Page 4, Type: table", user_prompt)

    @patch("RAG.query_engine.get_query_embedding", return_value=[0.1]*3072)
    @patch("RAG.query_engine.search_vectors")
    @patch("RAG.query_engine.ChatGoogleGenerativeAI")
    def test_02_multimodal_context_assembly_formatting(self, mock_chat, mock_search, mock_embed):
        """Verify specialized context assembly for text, chart, diagram, and tables."""
        # 1. Setup mock search hits covering all types
        mock_hits = [
            {"id": "chunk-text-1", "score": 0.95, "payload": {"document_id": self.doc_id, "page_number": 1, "chunk_type": "text"}},
            {"id": "chunk-chart-1", "score": 0.90, "payload": {"document_id": self.doc_id, "page_number": 2, "chunk_type": "image"}},
            {"id": "chunk-diagram-1", "score": 0.85, "payload": {"document_id": self.doc_id, "page_number": 3, "chunk_type": "image"}},
            {"id": "chunk-table-1", "score": 0.80, "payload": {"document_id": self.doc_id, "page_number": 4, "chunk_type": "table"}},
            {"id": "chunk-table-2", "score": 0.75, "payload": {"document_id": self.doc_id, "page_number": 5, "chunk_type": "table"}}
        ]
        mock_search.return_value = mock_hits

        mock_chat_inst = MagicMock()
        mock_chat_inst.invoke.return_value = AIMessage(content="Answer details.")
        mock_chat_inst.return_value = AIMessage(content="Answer details.")
        mock_chat.return_value = mock_chat_inst

        # 2. Run retrieval query with top_k=5
        execute_rag_query("Get all metrics", self.db_path, self.vector_db_path, top_k=5)

        call_args = mock_chat_inst.invoke.call_args or mock_chat_inst.call_args
        messages = call_args[0][0].messages
        user_prompt = messages[1].content if hasattr(messages[1], "content") else str(messages[1])

        # 3. Assert on correct custom context assembly parts (Phase 4):
        # A. Text: appends content normally
        self.assertIn("Standard paragraph introducing sensors.", user_prompt)

        # B. Chart: appends summary + insights
        self.assertIn("Chart Summary:\n# Revenue growth chart summary", user_prompt)
        self.assertIn("Extracted Insights:\n- Highest spike in Q2.\n- Consistent yearly uptrend.", user_prompt)

        # C. Diagram: appends relationships + workflow
        self.assertIn("Relationships:\n- Client -> Server (HTTPS)", user_prompt)
        self.assertIn("Workflow Description:\n- Initiate hand-shake\n- Authorize client token", user_prompt)

        # D. Financial Table: appends findings + stats
        self.assertIn("Table Findings:\n- OpEx is down 5%.", user_prompt)
        self.assertIn("Statistics:\n- Revenue: {'Q1': '$100K'}", user_prompt)

        # E. Statistical Table: appends findings + stats
        self.assertIn("Table Findings:\n- Treatment is statistically significant.", user_prompt)
        self.assertIn("Statistics:\n- Treatment: {'p-val': '<0.005'}", user_prompt)

    @patch("RAG.query_engine.get_query_embedding", return_value=[0.1]*3072)
    @patch("RAG.query_engine.search_vectors")
    @patch("RAG.query_engine.ChatGoogleGenerativeAI")
    def test_03_source_transparency_citations(self, mock_chat, mock_search, mock_embed):
        """Verify system instructions enforce new transparent citation schema."""
        mock_search.return_value = [
            {"id": "chunk-chart-1", "score": 0.90, "payload": {"document_id": self.doc_id, "page_number": 2, "chunk_type": "image"}}
        ]
        mock_chat_inst = MagicMock()
        mock_chat_inst.invoke.return_value = AIMessage(content="Sample response.")
        mock_chat_inst.return_value = AIMessage(content="Sample response.")
        mock_chat.return_value = mock_chat_inst

        execute_rag_query("Get opex chart", self.db_path, self.vector_db_path, top_k=1)

        call_args = mock_chat_inst.invoke.call_args or mock_chat_inst.call_args
        messages = call_args[0][0].messages
        sys_prompt = messages[0].content if hasattr(messages[0], "content") else str(messages[0])

        # Verify citation instruction is updated
        self.assertIn("[filename.pdf (Page X, Asset: asset_type, Sub-Type: classification_type)]", sys_prompt)
        self.assertIn("Cite the exact document, page number, asset type, and classification type", sys_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
