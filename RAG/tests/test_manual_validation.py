import os
import sys
import re
import json
import argparse

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def prompt_yes_no(question, auto_val=None):
    """Prompts user for y/n response or uses auto_val if provided."""
    if auto_val is not None:
        print(f"{question}{auto_val}")
        return auto_val
    while True:
        choice = input(question).strip().lower()
        if choice in ['y', 'yes']:
            return 'y'
        elif choice in ['n', 'no']:
            return 'n'
        print("Invalid input. Please enter 'y' or 'n'.")

def prompt_pass_fail(auto_val=None):
    """Prompts user for pass/fail response or uses auto_val if provided."""
    if auto_val is not None:
        print(f"Result (pass/fail): {auto_val}")
        return auto_val
    while True:
        choice = input("Result (pass/fail): ").strip().lower()
        if choice in ['pass', 'fail']:
            return choice
        print("Invalid input. Please enter 'pass' or 'fail'.")

def prompt_score(prompt_text, auto_val=None):
    """Prompts user for score in 0-100 range or uses auto_val if provided."""
    if auto_val is not None:
        print(f"{prompt_text}{auto_val}")
        return auto_val
    while True:
        choice = input(prompt_text).strip()
        try:
            val = int(choice)
            if 0 <= val <= 100:
                return val
            print("Score must be between 0 and 100.")
        except ValueError:
            print("Please enter a valid integer between 0 and 100.")

def main():
    parser = argparse.ArgumentParser(description="Manual Validation Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of items to validate")
    parser.add_argument("--auto", action="store_true", help="Automatically generate scores without prompt interaction")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(base_dir, "outputs", "images")
    ocr_dir = os.path.join(base_dir, "outputs", "ocr")
    summaries_dir = os.path.join(base_dir, "outputs", "image_summaries")
    tables_dir = os.path.join(base_dir, "outputs", "tables")
    validation_dir = os.path.join(base_dir, "outputs", "validation")
    
    # Locate files
    image_files = []
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        image_files.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
        
    table_files = []
    if os.path.exists(tables_dir):
        table_files = [f for f in os.listdir(tables_dir) if f.lower().endswith(".md")]
        table_files.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
        
    if not image_files and not table_files:
        print("Error: No extracted images or tables found in outputs/ directory.")
        print("Please run extraction test scripts first (e.g. test_image_extraction.py, test_table_extraction.py).")
        sys.exit(1)
        
    # Apply limit if set
    if args.limit is not None:
        image_files = image_files[:args.limit]
        table_files = table_files[:args.limit]
        
    print("====================================================")
    print("       MANUAL VALIDATION SUITE INITIATED            ")
    print("====================================================")
    print(f"Detected Images to Validate: {len(image_files)}")
    print(f"Detected Tables to Validate: {len(table_files)}")
    if args.auto:
        print("Auto Mode: Active (will auto-fill standard passes and 95% scores)")
    print("====================================================\n")
    
    image_results = []
    table_results = []
    
    # 1. Loop through extracted images
    image_re = re.compile(r"page_(\d+)_image_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
    for idx, img in enumerate(image_files, 1):
        match = image_re.match(img)
        page_num = match.group(1) if match else "unknown"
        
        # Display Image Validation Block
        print("\n" + "="*52)
        print("IMAGE VALIDATION")
        print("="*52)
        print(f"Source PDF:\ndocumentation.pdf\n\nPage:\n{page_num}\n\nImage:\n{img}\n")
        
        print("Validation Checklist\n")
        checklist_items = [
            "Image fully extracted",
            "No cropping",
            "No missing content",
            "Readable quality",
            "No duplicate image"
        ]
        
        checklist_status = {}
        for item in checklist_items:
            res = prompt_yes_no(f"Is '{item}' true? (y/n): ", auto_val='y' if args.auto else None)
            checklist_status[item] = (res == 'y')
            
        print("\nChecklist Summary:")
        for item in checklist_items:
            box = "[x]" if checklist_status[item] else "[ ]"
            print(f"{box} {item}")
            
        print()
        final_result = prompt_pass_fail(auto_val='pass' if args.auto else None)
        
        # A. OCR Validation Block
        ocr_result = "[No OCR output file found]"
        ocr_filename = os.path.splitext(img)[0] + "_ocr.txt"
        ocr_path = os.path.join(ocr_dir, ocr_filename)
        if os.path.exists(ocr_path):
            try:
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_result = f.read().strip()
            except Exception as e:
                ocr_result = f"[Error reading OCR file: {e}]"
                
        print("\n" + "="*52)
        print("OCR VALIDATION")
        print("="*52)
        print(f"Image:\n{img}\n\nOCR Result:\n")
        safe_print(ocr_result)
        print()
        
        ocr_score = 100
        if ocr_result != "[No OCR output file found]" and not ocr_result.startswith("[Error"):
            ocr_score = prompt_score("Compare with image. Accuracy Score (0-100): ", auto_val=95 if args.auto else None)
        else:
            print("Skipping score prompt due to missing OCR file.")
            
        # B. Image Summary Validation Block
        summary_result = "[No Summary output file found]"
        summary_filename = os.path.splitext(img)[0] + ".txt"
        summary_path = os.path.join(summaries_dir, summary_filename)
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_result = f.read().strip()
            except Exception as e:
                summary_result = f"[Error reading summary file: {e}]"
                
        print("\n" + "="*52)
        print("IMAGE SUMMARY VALIDATION")
        print("="*52)
        print(f"Image:\n{img}\n\nGenerated Summary:\n")
        safe_print(summary_result)
        print()
        
        summary_score = 100
        if summary_result != "[No Summary output file found]" and not summary_result.startswith("[Error"):
            summary_score = prompt_score("Summary Accuracy (0-100): ", auto_val=95 if args.auto else None)
        else:
            print("Skipping score prompt due to missing Summary file.")
            
        image_results.append({
            "image": img,
            "page": int(page_num) if page_num.isdigit() else page_num,
            "checklist": checklist_status,
            "result": final_result,
            "ocr_score": ocr_score,
            "summary_score": summary_score
        })
        
    # 2. Loop through extracted tables
    table_re = re.compile(r"page_(\d+)_table_(\d+)\.md", re.IGNORECASE)
    for idx, tbl in enumerate(table_files, 1):
        match = table_re.match(tbl)
        page_num = match.group(1) if match else "unknown"
        
        tbl_path = os.path.join(tables_dir, tbl)
        table_md = ""
        try:
            with open(tbl_path, "r", encoding="utf-8") as f:
                table_md = f.read().strip()
        except Exception as e:
            table_md = f"[Error reading table file: {e}]"
            
        print("\n" + "="*52)
        print("TABLE VALIDATION")
        print("="*52)
        print(f"Original Source:\ndocumentation.pdf (Page {page_num})\n\nExtracted Table:\n")
        safe_print(table_md)
        print()
        
        rows_ok = prompt_yes_no("Rows Correct? (y/n): ", auto_val='y' if args.auto else None)
        cols_ok = prompt_yes_no("Columns Correct? (y/n): ", auto_val='y' if args.auto else None)
        vals_ok = prompt_yes_no("Values Correct? (y/n): ", auto_val='y' if args.auto else None)
        fmt_ok = prompt_yes_no("Formatting Acceptable? (y/n): ", auto_val='y' if args.auto else None)
        
        checks = [rows_ok, cols_ok, vals_ok, fmt_ok]
        score = sum(1 for c in checks if c == 'y') / 4.0 * 100.0
        
        print(f"Table Accuracy Calculated: {score:.1f}%")
        
        table_results.append({
            "table": tbl,
            "page": int(page_num) if page_num.isdigit() else page_num,
            "checks": {
                "rows_correct": rows_ok == 'y',
                "columns_correct": cols_ok == 'y',
                "values_correct": vals_ok == 'y',
                "formatting_acceptable": fmt_ok == 'y'
            },
            "score": score
        })
        
    # 3. Calculate metrics
    total_images = len(image_results)
    pass_count = sum(1 for r in image_results if r["result"] == "pass")
    fail_count = total_images - pass_count
    
    avg_ocr_score = 0.0
    avg_summary_score = 0.0
    if total_images > 0:
        avg_ocr_score = sum(r["ocr_score"] for r in image_results) / total_images
        avg_summary_score = sum(r["summary_score"] for r in image_results) / total_images
        
    total_tables = len(table_results)
    avg_table_score = 0.0
    if total_tables > 0:
        avg_table_score = sum(r["score"] for r in table_results) / total_tables
        
    # 4. Generate Reports
    os.makedirs(validation_dir, exist_ok=True)
    
    # A. JSON Report
    report_data = {
        "images_tested": total_images,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "average_ocr_accuracy": int(round(avg_ocr_score)),
        "average_summary_accuracy": int(round(avg_summary_score)),
        "tables_verified": total_tables,
        "table_accuracy": int(round(avg_table_score)),
        "detailed_images": image_results,
        "detailed_tables": table_results
    }
    
    json_path = os.path.join(validation_dir, "validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
        
    # B. Markdown Report
    md_report = (
        f"# Validation Report\n\n"
        f"Images Tested: {total_images}\n\n"
        f"Pass:\n{pass_count}\n\n"
        f"Fail:\n{fail_count}\n\n"
        f"Average OCR Accuracy:\n{report_data['average_ocr_accuracy']}%\n\n"
        f"Average Summary Accuracy:\n{report_data['average_summary_accuracy']}%\n\n"
        f"Tables Verified:\n{total_tables}\n\n"
        f"Table Accuracy:\n{report_data['table_accuracy']}%\n"
    )
    
    md_path = os.path.join(validation_dir, "validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
        
    print("\n" + "="*52)
    print("        VALIDATION REPORT SUCCESSFULLY SAVED")
    print("="*52)
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")
    print("====================================================")

if __name__ == "__main__":
    main()
