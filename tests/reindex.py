"""
Re-indexing Script for Separate Collections Architecture

This script clears the existing vector store and re-indexes all PDFs
in the data/ folder with the new separate collections structure.

Use this when you've updated normalization rules or the VectorStore architecture.
"""

import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import shutil
from pathlib import Path
from Retrieval import get_vdb, get_file_hash, run_rag_pipeline

class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def clear_vector_store():
    """
    Clears the entire vector store (all collections).
    WARNING: This deletes all indexed data!
    """
    print(f"\n{Colors.WARNING}⚠ WARNING: This will delete all indexed documents!{Colors.ENDC}")
    confirm = input(f"{Colors.BOLD}Type 'YES' to confirm: {Colors.ENDC}")
    
    if confirm != 'YES':
        print(f"{Colors.FAIL}Aborted.{Colors.ENDC}")
        return False
    
    # Remove FAISS store directory (with new collections/ structure)
    faiss_path = Path("faiss_store")
    if faiss_path.exists():
        shutil.rmtree(faiss_path)
        print(f"{Colors.OKGREEN}✓ Cleared vector store (all collections){Colors.ENDC}")
    
    # Recreate structure
    (faiss_path / "collections").mkdir(parents=True, exist_ok=True)
    print(f"{Colors.OKGREEN}✓ Created new collections structure{Colors.ENDC}")
    
    return True


def scan_data_folder(data_dir="data"):
    """Scan for all PDFs in the data folder."""
    pdf_pattern = os.path.join(data_dir, "*.pdf")
    pdf_paths = glob.glob(pdf_pattern)
    return pdf_paths


def reindex_all_pdfs():
    """
    Re-index all PDFs in data/ folder with new separate collections.
    """
    data_dir = "data"
    print(f"\n{Colors.BOLD}Scanning for PDFs in {data_dir}/...{Colors.ENDC}")
    pdf_paths = scan_data_folder(data_dir)
    
    if not pdf_paths:
        print(f"{Colors.WARNING}No PDFs found in {data_dir}/ folder!{Colors.ENDC}")
        return
    
    print(f"{Colors.OKGREEN}Found {len(pdf_paths)} PDF(s){Colors.ENDC}\n")
    
    # Index each PDF with run_rag_pipeline (creates separate collection per PDF)
    hash_mapping = {}
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        filename = os.path.basename(pdf_path)
        print(f"\n{Colors.BOLD}[{i}/{len(pdf_paths)}] Processing: {filename}{Colors.ENDC}")
        
        try:
            # Get hash
            pdf_hash = get_file_hash(pdf_path)
            
            # Index (creates collection in faiss_store/collections/{hash}/)
            run_rag_pipeline(pdf_path, query="", top_k=1)
            
            # Store mapping
            hash_mapping[pdf_hash] = filename
            
            print(f"{Colors.OKGREEN}✓ Indexed {filename} -> {pdf_hash[:8]}...{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.FAIL}✗ Error indexing {filename}: {e}{Colors.ENDC}")
    
    # Save hash mapping
    save_hash_mapping(hash_mapping)
    
    print(f"\n{Colors.OKGREEN}✓ Re-indexing complete! {len(hash_mapping)} PDFs indexed.{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Each PDF now has its own collection in faiss_store/collections/{Colors.ENDC}")


def save_hash_mapping(hash_to_filename):
    """Save PDF hash to filename mapping."""
    import json
    mapping_file = "faiss_store/hash_to_filename.json"
    
    # Ensure directory exists
    os.makedirs("faiss_store", exist_ok=True)
    
    with open(mapping_file, 'w') as f:
        json.dump(hash_to_filename, f, indent=2)
    
    print(f"{Colors.OKGREEN}✓ Saved hash mapping{Colors.ENDC}")


def verify_collections():
    """
    Verify that collections were created correctly.
    """
    print(f"\n{Colors.BOLD}Verifying Collections...{Colors.ENDC}\n")
    
    collections_path = Path("faiss_store/collections")
    
    if not collections_path.exists():
        print(f"{Colors.WARNING}No collections directory found!{Colors.ENDC}")
        return
    
    # List all collection folders
    collections = [d for d in collections_path.iterdir() if d.is_dir()]
    
    if not collections:
        print(f"{Colors.WARNING}No collections found. Please index some documents first.{Colors.ENDC}")
        return
    
    print(f"Found {len(collections)} collection(s):\n")
    
    for i, coll_dir in enumerate(collections, 1):
        pdf_hash = coll_dir.name
        index_file = coll_dir / "index.faiss"
        metadata_file = coll_dir / "metadata.json"
        
        print(f"{Colors.BOLD}Collection {i}: {pdf_hash[:16]}...{Colors.ENDC}")
        
        # Check files
        if index_file.exists() and metadata_file.exists():
            # Load metadata to count chunks
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"  {Colors.OKGREEN}✓ index.faiss exists{Colors.ENDC}")
            print(f"  {Colors.OKGREEN}✓ metadata.json exists{Colors.ENDC}")
            print(f"  {Colors.OKGREEN}✓ {len(metadata)} chunks indexed{Colors.ENDC}")
        else:
            print(f"  {Colors.FAIL}✗ Incomplete collection (missing files){Colors.ENDC}")
        
        print()


def view_collection_stats():
    """Show statistics for all collections."""
    print(f"\n{Colors.BOLD}Collection Statistics{Colors.ENDC}\n")
    
    vdb = get_vdb()
    sources = vdb.get_all_sources()
    
    if not sources:
        print(f"{Colors.WARNING}No sources found!{Colors.ENDC}")
        return
    
    # Load hash mapping
    import json
    mapping_file = "faiss_store/hash_to_filename.json"
    hash_to_filename = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            hash_to_filename = json.load(f)
    
    print(f"Total Documents: {len(sources)}\n")
    
    for i, source in enumerate(sources, 1):
        pdf_hash = source['source_hash']
        filename = hash_to_filename.get(pdf_hash, "Unknown")
        chunk_count = source['chunk_count']
        
        print(f"{i}. {Colors.BOLD}{filename}{Colors.ENDC}")
        print(f"   Hash: {pdf_hash[:16]}...")
        print(f"   Chunks: {chunk_count}")
        print()


def main():
    """Main menu"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Re-indexing Tool (Separate Collections){Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    print("This tool manages the vector store with the new")
    print("separate collections architecture.\n")
    
    print(f"{Colors.BOLD}Options:{Colors.ENDC}")
    print("1. Clear all collections and re-index all PDFs (RECOMMENDED for migration)")
    print("2. Verify current collections")
    print("3. View collection statistics")
    print("4. Exit")
    
    choice = input(f"\n{Colors.BOLD}Select option (1-4): {Colors.ENDC}").strip()
    
    if choice == '1':
        if clear_vector_store():
            reindex_all_pdfs()
            verify_collections()
    elif choice == '2':
        verify_collections()
    elif choice == '3':
        view_collection_stats()
    elif choice == '4':
        print("\nExiting...")
    else:
        print(f"{Colors.FAIL}Invalid choice{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")
