"""
RAG/tests/test_knowledge_normalizer.py

Unit tests for the Knowledge Normalization Layer.
Verifies that various extractor output structures are correctly normalized into
NormalizedKnowledgeChunk schemas and that embedding_text is generated intelligently.
"""

import unittest
import sys
import os

# Ensure project root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from RAG.knowledge_normalizer import normalize_knowledge_chunk, NormalizedKnowledgeChunk


class TestKnowledgeNormalizer(unittest.TestCase):
    """Verifies schema structure, metadata mapping, and embedding_text formatting."""

    def test_01_text_image_normalization(self):
        kobj = {
            "cleaned_text": "Cleaned scanned text details.",
            "extracted_text": "Raw scanned text.",
            "word_count": 4
        }
        res = normalize_knowledge_chunk(
            document_name="scanned_doc.png",
            page_number=2,
            asset_type="image",
            classification_type="TEXT_IMAGE",
            extractor_used="extract_text_knowledge",
            knowledge_object=kobj,
            rich_text_representation="# Extracted Document Text\nCleaned scanned text details."
        )

        self.assertIsInstance(res, NormalizedKnowledgeChunk)
        self.assertEqual(res.document_name, "scanned_doc.png")
        self.assertEqual(res.page_number, 2)
        self.assertEqual(res.asset_type, "image")
        self.assertEqual(res.classification_type, "TEXT_IMAGE")
        self.assertEqual(res.extractor_used, "extract_text_knowledge")
        self.assertEqual(res.rich_text_representation, "# Extracted Document Text\nCleaned scanned text details.")
        self.assertEqual(res.structured_knowledge, kobj)
        self.assertIn("Cleaned scanned text details.", res.embedding_text)
        self.assertIsNotNone(res.chunk_id)
        self.assertEqual(res.metadata["document_name"], "scanned_doc.png")

    def test_02_chart_normalization(self):
        kobj = {
            "title": "Quarterly Revenue Growth",
            "chart_type": "bar_chart",
            "x_axis": "Quarter",
            "y_axis": "Revenue ($M)",
            "data_points": [{"label": "Q1", "value": "15"}, {"label": "Q2", "value": "18"}],
            "trends": ["Revenue shows upward trajectory."],
            "insights": ["Highest spike in Q2."]
        }
        res = normalize_knowledge_chunk(
            document_name="chart_report.pdf",
            page_number=5,
            asset_type="image",
            classification_type="CHART",
            extractor_used="extract_chart_knowledge",
            knowledge_object=kobj,
            rich_text_representation="# Chart Summary\nUpswing in Q2."
        )

        self.assertEqual(res.asset_type, "image")
        self.assertEqual(res.classification_type, "CHART")
        self.assertIn("Quarterly Revenue Growth", res.embedding_text)
        self.assertIn("Q1: 15, Q2: 18", res.embedding_text)
        self.assertIn("Revenue shows upward trajectory.", res.embedding_text)
        self.assertIn("Highest spike in Q2.", res.embedding_text)
        self.assertIn("Upswing in Q2.", res.embedding_text)

    def test_03_diagram_normalization(self):
        kobj = {
            "diagram_type": "architecture_diagram",
            "nodes": ["Client", "API Gateway", "Database"],
            "relationships": [
                {"from": "Client", "to": "API Gateway", "label_or_relationship": "HTTPS request"},
                {"from": "API Gateway", "to": "Database", "label_or_relationship": "gRPC"}
            ],
            "workflow": ["Client initiates call", "Gateway routes request"],
            "components": ["AWS VPC"],
            "summary": "Typical cloud architecture mapping."
        }
        res = normalize_knowledge_chunk(
            document_name="system_flow.png",
            page_number=1,
            asset_type="image",
            classification_type="DIAGRAM",
            extractor_used="extract_diagram_knowledge",
            knowledge_object=kobj,
            rich_text_representation="# Architecture Summary\nCloud flow details."
        )

        self.assertEqual(res.classification_type, "DIAGRAM")
        self.assertIn("Client -> API Gateway (HTTPS request)", res.embedding_text)
        self.assertIn("API Gateway -> Database (gRPC)", res.embedding_text)
        self.assertIn("Client, API Gateway, Database", res.embedding_text)
        self.assertIn("Cloud flow details.", res.embedding_text)

    def test_04_mixed_normalization(self):
        kobj = {
            "sections": ["Layout Section 1", "Layout Section 2"],
            "headings": ["Infographic Heading"],
            "labels": ["Stat annotation"],
            "process_flow": ["Step 1", "Step 2"],
            "key_takeaways": ["Takeaway metric description"],
            "summary": "Comprehensive infographic contents summary."
        }
        res = normalize_knowledge_chunk(
            document_name="infographic.png",
            page_number=1,
            asset_type="image",
            classification_type="MIXED",
            extractor_used="run_visual_understanding_logic",
            knowledge_object=kobj,
            rich_text_representation="# Infographic Layout\nVisual details."
        )

        self.assertEqual(res.classification_type, "MIXED")
        self.assertIn("Layout Section 1, Layout Section 2", res.embedding_text)
        self.assertIn("Infographic Heading", res.embedding_text)
        self.assertIn("Takeaway metric description", res.embedding_text)
        self.assertIn("Visual details.", res.embedding_text)

    def test_05_table_financial_normalization(self):
        kobj = {
            "title": "Income Statement",
            "reporting_period": "FY 2026",
            "currency": "EUR",
            "financial_metrics": [{"category": "Revenue", "values": {"FY26": "120M"}}],
            "key_financial_takeaways": ["Revenue up by 10%."],
            "summary": "Yearly profit assessment table."
        }
        res = normalize_knowledge_chunk(
            document_name="financials.xlsx",
            page_number=1,
            asset_type="table",
            classification_type="TABLE_FINANCIAL",
            extractor_used="extract_financial_table",
            knowledge_object=kobj,
            rich_text_representation="# Financial Balance Sheet\nTotal: 120M."
        )

        self.assertEqual(res.asset_type, "table")
        self.assertEqual(res.classification_type, "TABLE_FINANCIAL")
        self.assertIn("Income Statement", res.embedding_text)
        self.assertIn("Revenue: {'FY26': '120M'}", res.embedding_text)
        self.assertIn("Revenue up by 10%.", res.embedding_text)
        self.assertIn("Total: 120M.", res.embedding_text)

    def test_06_table_statistical_normalization(self):
        kobj = {
            "title": "A/B Test Results",
            "variables": ["Control", "Treatment"],
            "metrics": ["CTR", "Conversion Rate"],
            "data_summary": [{"variable_or_group": "Treatment", "metric_values": {"CTR": "0.15"}}],
            "statistical_conclusions": ["Treatment is statistically significant."],
            "summary": "Conversion statistics matrix."
        }
        res = normalize_knowledge_chunk(
            document_name="ab_test.csv",
            page_number=3,
            asset_type="table",
            classification_type="TABLE_STATISTICAL",
            extractor_used="extract_statistical_table",
            knowledge_object=kobj,
            rich_text_representation="# Statistical Report\nSignificant growth."
        )

        self.assertEqual(res.classification_type, "TABLE_STATISTICAL")
        self.assertIn("A/B Test Results", res.embedding_text)
        self.assertIn("CTR, Conversion Rate", res.embedding_text)
        self.assertIn("Treatment is statistically significant.", res.embedding_text)
        self.assertIn("Significant growth.", res.embedding_text)

    def test_07_table_timeseries_normalization(self):
        kobj = {
            "title": "CPU Usage Chronological",
            "time_interval": "hourly",
            "timestamps": ["12:00", "13:00"],
            "series_data": [{"metric_name": "CPU_Load", "values_over_time": ["40%", "45%"]}],
            "trends_observed": ["Spike at lunch time."],
            "summary": "Chronological load table."
        }
        res = normalize_knowledge_chunk(
            document_name="logs.csv",
            page_number=10,
            asset_type="table",
            classification_type="TABLE_TIMESERIES",
            extractor_used="extract_timeseries_table",
            knowledge_object=kobj,
            rich_text_representation="# CPU Logs\nPeak loads."
        )

        self.assertEqual(res.classification_type, "TABLE_TIMESERIES")
        self.assertIn("CPU Usage Chronological", res.embedding_text)
        self.assertIn("12:00, 13:00", res.embedding_text)
        self.assertIn("CPU_Load: ['40%', '45%']", res.embedding_text)
        self.assertIn("Spike at lunch time.", res.embedding_text)
        self.assertIn("Peak loads.", res.embedding_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
