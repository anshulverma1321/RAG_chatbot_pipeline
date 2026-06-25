import os
import sys
import unittest

# Ensure root folder is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app

class TestValidationAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.pdf_path = os.path.join(BASE_DIR, "RAG", "data", "test2.pdf")
        
        # Locate an image for testing
        cls.image_path = None
        images_dir = os.path.join(BASE_DIR, "outputs", "images")
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if images:
                cls.image_path = os.path.join(images_dir, images[0])
                
        # Write sample CSV
        cls.sample_csv_path = os.path.join(BASE_DIR, "RAG", "data", "temp", "test_sample.csv")
        os.makedirs(os.path.dirname(cls.sample_csv_path), exist_ok=True)
        with open(cls.sample_csv_path, "w", encoding="utf-8") as f:
            f.write("Name,Age,College\nJohn,22,MIT\nAlice,24,Stanford\nBob,23,Harvard\n")
            
        # Write sample Excel
        cls.sample_xlsx_path = os.path.join(BASE_DIR, "RAG", "data", "temp", "test_sample.xlsx")
        import pandas as pd
        df = pd.DataFrame([
            {"Name": "John", "Age": 22, "College": "MIT"},
            {"Name": "Alice", "Age": 24, "College": "Stanford"},
            {"Name": "Bob", "Age": 23, "College": "Harvard"}
        ])
        df.to_excel(cls.sample_xlsx_path, index=False)
        
        # Write empty file
        cls.empty_file_path = os.path.join(BASE_DIR, "RAG", "data", "temp", "test_empty.csv")
        with open(cls.empty_file_path, "wb") as f:
            pass
            
        # Write corrupted Excel file
        cls.corrupted_xlsx_path = os.path.join(BASE_DIR, "RAG", "data", "temp", "test_corrupt.xlsx")
        with open(cls.corrupted_xlsx_path, "wb") as f:
            f.write(b"this is corrupt xlsx data, not a real zip file!")

    @classmethod
    def tearDownClass(cls):
        for path in [cls.sample_csv_path, cls.sample_xlsx_path, cls.empty_file_path, cls.corrupted_xlsx_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_image_summary_endpoint(self):
        if not self.image_path:
            self.skipTest("No test image found in outputs/images")
            
        print(f"\n--- Testing POST /validation/image-summary with {os.path.basename(self.image_path)} ---")
        with open(self.image_path, "rb") as f:
            files = {"file": (os.path.basename(self.image_path), f, "image/jpeg")}
            response = self.client.post("/validation/image-summary?debug=true", files=files)
            
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("filename", data)
        self.assertIn("summary", data)
        self.assertIn("processing_time", data)
        self.assertEqual(data["status"], "success")
        print("Response Summary:", data["summary"][:100], "...")
        print("Processing Time:", data["processing_time"], "seconds")

    def test_tabular_data_analysis_pdf(self):
        if not os.path.exists(self.pdf_path):
            self.skipTest("PDF path not found")
        print(f"\n--- Testing POST /validation/tabular-data-analysis with PDF ---")
        with open(self.pdf_path, "rb") as f:
            files = {"file": (os.path.basename(self.pdf_path), f, "application/pdf")}
            response = self.client.post("/validation/tabular-data-analysis?debug=true", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["detected_file_type"], "pdf")
        self.assertIn("tables_found", data)
        self.assertIn("tables", data)
        self.assertIn("preview", data)
        self.assertIn("file_size_kb", data)
        print("PDF Tables Found:", data["tables_found"])
        print("PDF Preview Rows:", len(data["preview"]) if data["preview"] else 0)

    def test_tabular_data_analysis_csv(self):
        print(f"\n--- Testing POST /validation/tabular-data-analysis with CSV ---")
        with open(self.sample_csv_path, "rb") as f:
            files = {"file": ("test_sample.csv", f, "text/csv")}
            response = self.client.post("/validation/tabular-data-analysis?debug=true", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["detected_file_type"], "csv")
        self.assertEqual(data["rows"], 3)
        self.assertEqual(data["columns"], 3)
        self.assertEqual(data["column_names"], ["Name", "Age", "College"])
        self.assertEqual(len(data["preview"]), 3)
        self.assertEqual(data["preview"][0]["Name"], "John")
        self.assertIn("file_size_kb", data)
        print("CSV Preview:", data["preview"])

    def test_tabular_data_analysis_xlsx(self):
        print(f"\n--- Testing POST /validation/tabular-data-analysis with XLSX ---")
        with open(self.sample_xlsx_path, "rb") as f:
            files = {"file": ("test_sample.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            response = self.client.post("/validation/tabular-data-analysis?debug=true", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["detected_file_type"], "xlsx")
        self.assertEqual(len(data["sheets"]), 1)
        self.assertEqual(data["sheets"][0]["sheet_name"], "Sheet1")
        self.assertEqual(data["sheets"][0]["rows"], 3)
        self.assertEqual(data["sheets"][0]["columns"], 3)
        self.assertEqual(data["sheets"][0]["column_names"], ["Name", "Age", "College"])
        self.assertEqual(len(data["preview"]), 3)
        self.assertEqual(data["preview"][0]["Name"], "John")
        self.assertIn("file_size_kb", data)
        print("Excel Sheets Info:", data["sheets"])

    def test_tabular_data_analysis_unsupported(self):
        print(f"\n--- Testing POST /validation/tabular-data-analysis with Unsupported format ---")
        # Use image_path if available as a dummy unsupported file for tabular-data-analysis
        test_file = self.image_path if self.image_path else self.pdf_path
        with open(test_file, "rb") as f:
            files = {"file": ("dummy.jpg", f, "image/jpeg")}
            response = self.client.post("/validation/tabular-data-analysis", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Unsupported file type", data["detail"])
        print("Unsupported File response detail:", data["detail"])

    def test_tabular_data_analysis_empty(self):
        print(f"\n--- Testing POST /validation/tabular-data-analysis with Empty file ---")
        with open(self.empty_file_path, "rb") as f:
            files = {"file": ("test_empty.csv", f, "text/csv")}
            response = self.client.post("/validation/tabular-data-analysis", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Empty file uploaded", data["detail"])
        print("Empty File response detail:", data["detail"])

    def test_tabular_data_analysis_corrupted(self):
        print(f"\n--- Testing POST /validation/tabular-data-analysis with Corrupted Excel ---")
        with open(self.corrupted_xlsx_path, "rb") as f:
            files = {"file": ("test_corrupt.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            response = self.client.post("/validation/tabular-data-analysis", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Corrupted or invalid Excel file", data["detail"])
        print("Corrupted Excel response detail:", data["detail"])

    def test_text_knowledge_extraction_endpoint(self):
        if not self.image_path:
            self.skipTest("No test image found")
        print(f"\n--- Testing POST /validation/text-knowledge-extraction ---")
        with open(self.image_path, "rb") as f:
            files = {"file": (os.path.basename(self.image_path), f, "image/jpeg")}
            response = self.client.post("/validation/text-knowledge-extraction", files=files)
        
        if response.status_code == 400:
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("error", data)
            print("OCR engine not functional in this environment. Endpoint returned correct error:", data["error"])
        else:
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["image_type"], "text_image")
            self.assertIn("extracted_text", data)
            self.assertIn("word_count", data)
            self.assertIn("rich_text_representation", data)
            self.assertIn("ocr_engine_used", data)
            self.assertIn("ocr_available", data)
            self.assertIn("ocr_raw_text", data)
            self.assertIn("ocr_text_length", data)
            self.assertIn("ocr_blocks_detected", data)
            self.assertIn("average_confidence", data)
            self.assertIn("cleaned_text", data)
            print("Text word count:", data["word_count"])
            print("OCR Engine used:", data["ocr_engine_used"])

    def test_chart_knowledge_extraction_endpoint(self):
        if not self.image_path:
            self.skipTest("No test image found")
        print(f"\n--- Testing POST /validation/chart-knowledge-extraction ---")
        with open(self.image_path, "rb") as f:
            files = {"file": (os.path.basename(self.image_path), f, "image/jpeg")}
            response = self.client.post("/validation/chart-knowledge-extraction", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["image_type"], "chart")
        self.assertIn("chart_type", data)
        self.assertIn("x_axis", data)
        self.assertIn("y_axis", data)
        self.assertIn("data_points", data)
        self.assertIn("insights", data)
        self.assertIn("rich_text_representation", data)
        print("Chart Type:", data["chart_type"])
        print("Insights extracted count:", len(data["insights"]))

    def test_diagram_knowledge_extraction_endpoint(self):
        if not self.image_path:
            self.skipTest("No test image found")
        print(f"\n--- Testing POST /validation/diagram-knowledge-extraction ---")
        with open(self.image_path, "rb") as f:
            files = {"file": (os.path.basename(self.image_path), f, "image/jpeg")}
            response = self.client.post("/validation/diagram-knowledge-extraction", files=files)
        
        if response.status_code == 400:
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("error", data)
            print("OCR engine not functional in this environment. Endpoint returned correct error:", data["error"])
        else:
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["image_type"], "diagram")
            self.assertIn("diagram_type", data)
            self.assertIn("nodes", data)
            self.assertIn("edges", data)
            self.assertIn("description", data)
            self.assertIn("rich_text_representation", data)
            print("Diagram Type:", data["diagram_type"])
            print("Nodes count:", len(data["nodes"]))
            print("Edges count:", len(data["edges"]))

if __name__ == "__main__":
    unittest.main()
