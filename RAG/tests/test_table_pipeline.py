import os
import sys
import json
import time
import pandas as pd

# Ensure root folder is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from RAG.app import app

def main():
    client = TestClient(app)
    
    # Create temp directory for validation assets if needed
    temp_dir = os.path.join(BASE_DIR, "RAG", "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # -----------------------------------------------------------------------
    # 1. Generate Mock Table Files Programmatically
    # -----------------------------------------------------------------------
    
    simple_csv_path = os.path.join(temp_dir, "test_simple.csv")
    df_simple = pd.DataFrame([
        {"First Name": "John", "Last Name": "Doe", "Department": "Engineering", "City": "Boston"},
        {"First Name": "Alice", "Last Name": "Smith", "Department": "Marketing", "City": "Chicago"},
        {"First Name": "Bob", "Last Name": "Johnson", "Department": "HR", "City": "Denver"}
    ])
    df_simple.to_csv(simple_csv_path, index=False)
    
    comparison_md_path = os.path.join(temp_dir, "test_comparison.md")
    comparison_content = (
        "| Model | Precision | Recall | Inference Speed | Parameters | Key Use Case |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| YOLOv8 | 0.925 | 0.880 | 12.4 ms | 3.2 M | Real-time object tracking in high fps video |\n"
        "| YOLOX | 0.941 | 0.908 | 15.1 ms | 9.0 M | Precision-focused static document inspection |\n"
        "| YOLOv7 | 0.912 | 0.865 | 11.0 ms | 6.2 M | Lightweight deployment on edge hardware |\n"
    )
    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write(comparison_content)
        
    financial_xlsx_path = os.path.join(temp_dir, "test_financial.xlsx")
    df_financial = pd.DataFrame([
        {"Financial Category": "Gross Revenue", "FY2022": "$145.2M", "FY2023": "$168.5M", "FY2024 (Projected)": "$195.0M"},
        {"Financial Category": "Cost of Goods Sold (COGS)", "FY2022": "$58.1M", "FY2023": "$64.0M", "FY2024 (Projected)": "$72.5M"},
        {"Financial Category": "Operating Expenses", "FY2022": "$42.0M", "FY2023": "$48.5M", "FY2024 (Projected)": "$53.0M"},
        {"Financial Category": "Net Operating Income", "FY2022": "$45.1M", "FY2023": "$56.0M", "FY2024 (Projected)": "$69.5M"}
    ])
    df_financial.to_excel(financial_xlsx_path, index=False)
    
    statistical_csv_path = os.path.join(temp_dir, "test_statistical.csv")
    df_stat = pd.DataFrame([
        {"Experimental Variable": "Baseline Model A", "Sample Count (N)": 150, "Mean Score": 0.784, "Standard Deviation (SD)": 0.045, "P-value": "Reference"},
        {"Experimental Variable": "Enhanced Model B (L1 regularization)", "Sample Count (N)": 150, "Mean Score": 0.812, "Standard Deviation (SD)": 0.038, "P-value": "0.024 (Significant)"},
        {"Experimental Variable": "Enhanced Model C (L2 weight decay)", "Sample Count (N)": 150, "Mean Score": 0.835, "Standard Deviation (SD)": 0.031, "P-value": "<0.001 (Highly Significant)"}
    ])
    df_stat.to_csv(statistical_csv_path, index=False)
    
    timeseries_xlsx_path = os.path.join(temp_dir, "test_timeseries.xlsx")
    df_ts = pd.DataFrame([
        {"Chronological Month": "January 2025", "Monthly Active Users": 82000, "Customer Acquisition Cost (CAC)": "$42.50", "Churn Rate": "4.2%"},
        {"Chronological Month": "February 2025", "Monthly Active Users": 85500, "Customer Acquisition Cost (CAC)": "$40.10", "Churn Rate": "4.0%"},
        {"Chronological Month": "March 2025", "Monthly Active Users": 91200, "Customer Acquisition Cost (CAC)": "$38.50", "Churn Rate": "3.8%"},
        {"Chronological Month": "April 2025", "Monthly Active Users": 98400, "Customer Acquisition Cost (CAC)": "$36.00", "Churn Rate": "3.5%"},
        {"Chronological Month": "May 2025", "Monthly Active Users": 105000, "Customer Acquisition Cost (CAC)": "$35.20", "Churn Rate": "3.2%"}
    ])
    df_ts.to_excel(timeseries_xlsx_path, index=False)

    test_files = {
        "test_simple.csv": ("TABLE_SIMPLE", "text/csv"),
        "test_comparison.md": ("TABLE_COMPARISON", "text/markdown"),
        "test_financial.xlsx": ("TABLE_FINANCIAL", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "test_statistical.csv": ("TABLE_STATISTICAL", "text/csv"),
        "test_timeseries.xlsx": ("TABLE_TIMESERIES", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }

    print("=" * 60)
    print("      NEW TABLE INTELLIGENCE ROUTING & VALIDATION")
    print("=" * 60)

    # 1. Test POST /validation/table-classify
    print("\n--- 1. Testing POST /validation/table-classify ---")
    for fname, (expected_type, mime) in test_files.items():
        fpath = os.path.join(temp_dir, fname)
        with open(fpath, "rb") as f:
            files = {"file": (fname, f, mime)}
            response = client.post("/validation/table-classify", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"File: {fname:<20} | Classified: {data.get('table_type'):<18} | Expected: {expected_type:<18} | Confidence: {data.get('confidence'):.4f}")
        else:
            print(f"File: {fname:<20} | Classification failed: {response.status_code} - {response.text}")

    # 2. Test POST /validation/table
    print("\n--- 2. Testing POST /validation/table ---")
    for fname, (expected_type, mime) in test_files.items():
        fpath = os.path.join(temp_dir, fname)
        print(f"\nProcessing single upload for {fname}...")
        t0 = time.perf_counter()
        with open(fpath, "rb") as f:
            files = {"file": (fname, f, mime)}
            response = client.post("/validation/table", files=files)
        elapsed = time.perf_counter() - t0
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Classification : {data.get('table_type')} (Confidence: {data.get('confidence'):.2f})")
            print(f"  Extractor Used : {data.get('extractor_selected')}")
            print(f"  Knowledge Keys : {list(data.get('knowledge', {}).keys())}")
            print(f"  Timings        : {data.get('timings')}")
            print(f"  Request Time   : {elapsed:.2f}s")
            
            # Print rich text markdown representation preview
            rich_text = data.get("rich_text_representation", "")
            preview = rich_text.split('\n')[:6]
            print("  Rich Text Preview:")
            print("\n".join(["    " + line for line in preview]))
        else:
            print(f"  Failed: {response.status_code} - {response.text}")

    # 3. Test POST /validation/table-debug
    print("\n--- 3. Testing POST /validation/table-debug ---")
    debug_file = "test_comparison.md"
    fpath = os.path.join(temp_dir, debug_file)
    with open(fpath, "rb") as f:
        files = {"file": (debug_file, f, "text/markdown")}
        response = client.post("/validation/table-debug", files=files)
        
    if response.status_code == 200:
        data = response.json()
        print(f"Debug classification keys : {list(data.get('classification', {}).keys())}")
        print(f"Selected route             : {data.get('selected_route')}")
        print(f"Raw extractor keys         : {list(data.get('raw_extractor_output', {}).keys())}")
        print(f"Debug timings              : {data.get('timings')}")
    else:
        print(f"Debug endpoint failed: {response.status_code} - {response.text}")

    # Clean up mock files
    print("\nCleaning up mock validation files...")
    for fname in test_files:
        fpath = os.path.join(temp_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    print("Cleanup done.")
    print("\n" + "=" * 60)
    print("  Table Pipeline test completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
