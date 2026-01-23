"""
Retrieval.py: Orchestrates ingestion, OCR, chunking, enrichment, and retrieval.
Manages interaction with the VectorDB.
"""

import hashlib
import os
import math
import numpy as np
from typing import List, Dict, Optional

from Normal import run_normal_pipeline
from Clean import run_clean_pipeline
from VectorStore import VectorDB

# Lazy initialization to avoid loading model at import time
_vdb = None

def get_vdb() -> VectorDB:
    """Gets or creates the VectorDB singleton."""
    global _vdb
    if _vdb is None:
        print("Initializing VectorDB...")
        _vdb = VectorDB()
    return _vdb

def get_file_hash(filepath: str) -> str:
    """MD5 hash of file content to track changes."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# -------------------------------------------------------
# INDEX BUILDER (No metadata enrichment needed)
# -------------------------------------------------------


# -------------------------------------------------------
# RETRIEVAL facade
# -------------------------------------------------------

def retrieve(
    query: str,
    filters: Optional[Dict] = None, # Metadata filters!
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieves top-k relevant chunks from Vector DB.
    """
    return get_vdb().query(query, n_results=top_k, where_filter=filters)


# -------------------------------------------------------
# RAG PIPELINE ENTRY
# -------------------------------------------------------

def run_rag_pipeline(pdf_path: str, query: str = "", top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Full pipeline with Persistence check.
    If query is empty, just ensures indexing is done.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Check if file is already processed
    file_hash = get_file_hash(pdf_path)
    
    vdb = get_vdb()  # Get singleton instance
    
    if vdb.check_processed(file_hash):
        print(f"Skipping OCR. Document {os.path.basename(pdf_path)} already in VectorDB.")
    else:
        print(f"New document detected (Hash: {file_hash}). Running OCR & Indexing...")
        
        # A. Clean + Chunk
        chunks = run_clean_pipeline(pdf_path)
        chunks = run_normal_pipeline(chunks) 
        
        # B. Ensure embedding_text exists (fallback for legacy)
        for chunk in chunks:
            if "embedding_text" not in chunk:
                if chunk.get("chunk_type") == "table_generic":
                    rows = chunk.get("table", {}).get("rows", [])
                    chunk["embedding_text"] = f"Table with {len(rows)} rows."
                else:
                    chunk["embedding_text"] = chunk.get("content", "")
        
        # C. Index
        vdb.add_chunks(chunks, file_hash)
    
    # 2. Retrieve (only if query provided)
    if query:
        results = retrieve(query, filters=filters, top_k=top_k)
        return results
    
    return []


# -------------------------------------------------------
# TEST ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH = "data/A_New_Algorithm_for_the_Determinant_and.pdf"

    query = "What is the determinant of the matrix?"

    results = run_rag_pipeline(PDF_PATH, query=query)

    print("\n" + "#" * 60)
    print("RETRIEVAL RESULTS")
    print("#" * 60)

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print(" Chunk ID :", r["chunk_id"])
        print(" Type     :", r["chunk_type"])
        print(" Page     :", r["page"])
        print(" Text     :", r["text"])

        if r["chunk_type"] == "table_generic":
            print(" Rows     :")
            for row in r["raw_chunk"]["table"]["rows"]:
                print(row)  
