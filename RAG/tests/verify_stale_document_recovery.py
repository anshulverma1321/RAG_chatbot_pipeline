import os
import sys
import hashlib
import sqlite3
import time
import requests

DB_PATH = "RAG/data/rag_tool.db"
TEST_FILE = "test_stale_recovery.csv"
SERVER_URL = "http://127.0.0.1:8000"

def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    # 1. Create a dummy CSV file
    csv_content = "Month,Revenue,Expenses\nJanuary,10000,5000\nFebruary,12000,6000\n"
    file_bytes = csv_content.encode("utf-8")
    file_hash = compute_hash(file_bytes)
    
    with open(TEST_FILE, "wb") as f:
        f.write(file_bytes)
        
    print(f"Created test file: {TEST_FILE}")
    print(f"Hash: {file_hash}")
    
    # 2. Insert stale document record with chunk_count = 0 in SQLite directly
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Clean up pre-existing records for this hash first
    cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
    row = cursor.fetchone()
    if row:
        doc_id = row[0]
        print(f"Cleaning up pre-existing records for document ID {doc_id}...")
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        cursor.execute("DELETE FROM ingestion_jobs WHERE file_hash = ?", (file_hash,))
        conn.commit()
        
    cursor.execute(
        "INSERT INTO documents (filename, file_hash, total_pages) VALUES (?, ?, ?)",
        (TEST_FILE, file_hash, 1)
    )
    conn.commit()
    stale_doc_id = cursor.lastrowid
    print(f"Created stale document record: ID={stale_doc_id}, filename={TEST_FILE}, chunk_count=0")
    
    # Double check chunks count is indeed 0
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (stale_doc_id,))
    stale_chunks = cursor.fetchone()[0]
    print(f"Verified stale document chunk count: {stale_chunks}")
    assert stale_chunks == 0, "Error: Stale document should have 0 chunks"
    conn.close()
    
    # 3. Perform duplicate upload of the test file to the running server
    upload_url = f"{SERVER_URL}/upload"
    print(f"Uploading duplicate file to {upload_url}...")
    with open(TEST_FILE, "rb") as f:
        response = requests.post(upload_url, files={"file": (TEST_FILE, f, "text/csv")})
        
    print("Upload Response Status Code:", response.status_code)
    resp_body = response.json()
    print("Upload Response Body:", resp_body)
    
    # Assertions on Upload response
    assert response.status_code == 202, f"Expected 202 but got {response.status_code}"
    assert resp_body.get("status") == "accepted", f"Expected 'accepted' status but got {resp_body.get('status')}"
    
    job_id = resp_body.get("job_id")
    print(f"Successfully triggered re-ingestion job: {job_id}")
    
    # 4. Poll job status from the running server until completion
    print("Polling job status...")
    success = False
    for i in range(30):
        job_resp = requests.get(f"{SERVER_URL}/jobs/{job_id}")
        job_body = job_resp.json()
        status = job_body.get("status")
        print(f"Attempt {i+1}: status={status}")
        if status == "success":
            success = True
            break
        elif status == "failed":
            print(f"Job failed with error: {job_body.get('error_message')}")
            break
        time.sleep(1)
        
    assert success, "Ingestion job did not complete successfully"
    
    # 5. Verify SQL DB after ingestion completes
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Verify stale document ID is deleted
    cursor.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (stale_doc_id,))
    stale_doc_exists = cursor.fetchone()[0]
    print(f"Stale document {stale_doc_id} exists count: {stale_doc_exists}")
    assert stale_doc_exists == 0, "Error: Stale document record was not deleted"
    
    # Verify new document is created with a new ID
    cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
    new_doc_row = cursor.fetchone()
    assert new_doc_row is not None, "Error: New document record not found"
    new_doc_id = new_doc_row[0]
    print(f"New document record created: ID={new_doc_id}")
    assert new_doc_id != stale_doc_id, "Error: New document ID should be different from stale document ID"
    
    # Verify new document has chunks > 0
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (new_doc_id,))
    new_chunks_count = cursor.fetchone()[0]
    print(f"New document chunk count: {new_chunks_count}")
    assert new_chunks_count > 0, f"Error: New document has 0 chunks"
    
    conn.close()
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    
    # Cleanup local test file
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
        print("Cleaned up local test file.")

if __name__ == "__main__":
    main()
