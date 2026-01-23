"""
VectorStore.py: FAISS-based vector database with separate collections per PDF.
Each PDF gets its own isolated index for clean deletion and management.
"""

import faiss
import numpy as np
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from Embeddings import LocalEmbedding


class VectorDB:
    def __init__(self, base_path: str = "faiss_store"):
        """
        Initialize VectorDB with separate collections per PDF.
        
        Structure:
        faiss_store/
        ├── collections/
        │   ├── {pdf_hash_1}/
        │   │   ├── index.faiss
        │   │   └── metadata.json
        │   ├── {pdf_hash_2}/
        │   │   ├── index.faiss
        │   │   └── metadata.json
        └── hash_to_filename.json (managed by app.py)
        """
        self.base_path = Path(base_path)
        self.collections_path = self.base_path / "collections"
        self.collections_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = 768
        self.embedding_fn = LocalEmbedding()
        
        print(f"VectorDB initialized with collections at: {self.collections_path}")
    
    def _get_collection_path(self, pdf_hash: str) -> Path:
        """Get the directory path for a specific PDF's collection."""
        return self.collections_path / pdf_hash
    
    def _load_collection(self, pdf_hash: str) -> Optional[Dict]:
        """
        Load a specific PDF's collection (index + metadata).
        Returns None if collection doesn't exist.
        """
        coll_path = self._get_collection_path(pdf_hash)
        
        if not coll_path.exists():
            return None
        
        index_file = coll_path / "index.faiss"
        metadata_file = coll_path / "metadata.json"
        
        if not index_file.exists() or not metadata_file.exists():
            print(f"Warning: Incomplete collection for {pdf_hash}")
            return None
        
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                'index': index,
                'metadata': metadata,
                'pdf_hash': pdf_hash
            }
        except Exception as e:
            print(f"Error loading collection {pdf_hash}: {e}")
            return None
    
    def _get_all_collection_hashes(self) -> List[str]:
        """Get list of all PDF hashes that have collections."""
        hashes = []
        
        if not self.collections_path.exists():
            return hashes
        
        for coll_dir in self.collections_path.iterdir():
            if coll_dir.is_dir():
                hashes.append(coll_dir.name)
        
        return hashes
    
    def add_chunks(self, chunks: List[Dict], pdf_hash: str):
        """
        Add chunks to a PDF-specific collection.
        Creates a new collection if it doesn't exist.
        """
        if not chunks:
            print("No chunks to add.")
            return
        
        print(f"Creating collection for PDF: {pdf_hash} with {len(chunks)} chunks")
        
        # Create collection directory
        coll_path = self._get_collection_path(pdf_hash)
        coll_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        documents = []
        metadata_list = []
        
        for chunk in chunks:
            # Embedding text
            documents.append(chunk["embedding_text"])
            
            # Optimized Metadata - Store only what's needed
            # For tables: store rows and essential fields
            # For text: store content and essential fields
            # Don't duplicate chunk_id, source_hash, chunk_type, etc.
            
            if chunk["chunk_type"] == "table_generic":
                # For tables, store only the rows (not the entire chunk)
                table_rows = chunk.get("table", {}).get("rows", [])
                content_data = json.dumps(table_rows)  # Just the rows
            else:
                # For text chunks, store just the content
                content_data = chunk.get("content", "")
            
            meta = {
                "chunk_id": chunk["chunk_id"],
                "source_hash": pdf_hash,
                "chunk_type": chunk["chunk_type"],
                "page": chunk.get("page", 0),
                "sections": json.dumps(chunk.get("sections", [])),
                "content": content_data,  # Only essential content
                "embedding_text": chunk["embedding_text"]  # For reference
            }
            metadata_list.append(meta)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_fn.embed_documents(documents)
        embeddings = np.array(embeddings, dtype='float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create new FAISS index for this PDF
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        
        # Save index
        index_file = coll_path / "index.faiss"
        faiss.write_index(index, str(index_file))
        
        # Save metadata
        metadata_file = coll_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"Collection saved: {len(chunks)} chunks indexed for {pdf_hash}")
    
    def query(self, query_text: str, n_results: int = 5, where_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search across all PDF collections (or a specific one if filtered).
        """
        # Determine which collections to search
        if where_filter and 'source_hash' in where_filter:
            # Search only specific PDF
            pdf_hashes = [where_filter['source_hash']]
        else:
            # Search all PDFs
            pdf_hashes = self._get_all_collection_hashes()
        
        if not pdf_hashes:
            print("No collections found in database")
            return []
        
        print(f"Searching {len(pdf_hashes)} collection(s) for: '{query_text}'")
        
        # Generate query embedding once
        query_embedding = self.embedding_fn.embed_query(query_text)
        query_embedding = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        # Search each collection and collect results
        all_results = []
        
        for pdf_hash in pdf_hashes:
            coll = self._load_collection(pdf_hash)
            if not coll:
                continue
            
            # Search this collection's index
            k = min(coll['index'].ntotal, n_results * 2)  # Get more for merging
            distances, indices = coll['index'].search(query_embedding, k)
            
            # Convert to results
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(coll['metadata']):
                    continue
                
                meta = coll['metadata'][idx]
                
                # Apply additional filters (e.g., intent)
                if where_filter:
                    match = True
                    for key, value in where_filter.items():
                        if key == 'source_hash':  # Already filtered
                            continue
                        if meta.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Reconstruct chunk from optimized metadata
                if meta["chunk_type"] == "table_generic":
                    table_rows = json.loads(meta["content"])
                    raw_chunk = {
                        "chunk_id": meta["chunk_id"],
                        "chunk_type": "table_generic",
                        "page": meta["page"],
                        "sections": json.loads(meta.get("sections", "[]")),
                        "table": {"rows": table_rows},
                        "embedding_text": meta["embedding_text"]
                    }
                else:
                    raw_chunk = {
                        "chunk_id": meta["chunk_id"],
                        "chunk_type": meta["chunk_type"],
                        "page": meta["page"],
                        "content": meta["content"],
                        "embedding_text": meta["embedding_text"]
                    }
                
                result = {
                    "chunk_id": meta["chunk_id"],
                    "source_hash": meta["source_hash"],
                    "chunk_type": meta["chunk_type"],
                    "page": meta["page"],
                    "text": meta["embedding_text"],
                    "raw_chunk": raw_chunk,
                    "distance": float(dist)
                }
                all_results.append(result)
        
        # Sort all results by distance (highest first for IP index)
        all_results.sort(key=lambda x: x['distance'], reverse=True)
        
        # Return top n_results
        return all_results[:n_results]
    
    def check_processed(self, pdf_hash: str) -> bool:
        """Check if a PDF has been indexed."""
        return self._get_collection_path(pdf_hash).exists()
    
    def delete_source(self, pdf_hash: str):
        """
        Delete a PDF collection - just remove the folder!
        This is clean and efficient with separate collections.
        """
        coll_path = self._get_collection_path(pdf_hash)
        
        if coll_path.exists():
            shutil.rmtree(coll_path)
            print(f"✓ Deleted collection: {pdf_hash}")
        else:
            print(f"Collection not found: {pdf_hash}")
    
    def get_all_sources(self) -> List[Dict[str, Any]]:
        """
        List all indexed PDFs with their chunk counts.
        """
        sources = []
        
        for pdf_hash in self._get_all_collection_hashes():
            coll = self._load_collection(pdf_hash)
            if coll:
                sources.append({
                    'source_hash': pdf_hash,
                    'chunk_count': len(coll['metadata'])
                })
        
        return sources


# Singleton instance
_vectordb_instance = None

def get_vector_db() -> VectorDB:
    """Get singleton VectorDB instance."""
    global _vectordb_instance
    if _vectordb_instance is None:
        _vectordb_instance = VectorDB()
    return _vectordb_instance
