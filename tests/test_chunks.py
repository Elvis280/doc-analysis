"""
Test/Inspection Tool for FAISS Vector Store (Collections-Based)
Updated for new separate collections architecture
"""

import json
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from Retrieval import get_vdb

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_separator(char='=', length=80):
    """Print separator line"""
    print(f"{Colors.OKBLUE}{char * length}{Colors.ENDC}")


def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")


def inspect_vector_store():
    """Main inspection function - updated for collections"""
    print_separator()
    print(f"{Colors.BOLD}{Colors.HEADER}{'FAISS Vector Store Inspection (Collections)'.center(80)}{Colors.ENDC}")
    print_separator()
    
    # Load VectorDB
    print("\nLoading Vector Store...")
    vdb = get_vdb()
    
    # Get all sources (collections)
    sources = vdb.get_all_sources()
    total_sources = len(sources)
    total_chunks = sum(s['chunk_count'] for s in sources)
    
    print(f"{Colors.OKGREEN}✓ Loaded successfully{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  Total chunks: {Colors.OKGREEN}{total_chunks}{Colors.ENDC}")
    print(f"  Total collections: {Colors.OKGREEN}{total_sources}{Colors.ENDC}")
    
    if total_sources == 0:
        print(f"\n{Colors.WARNING}No collections found. Upload some PDFs first.{Colors.ENDC}")
        return
    
    # Load filename mapping
    try:
        with open('faiss_store/hash_to_filename.json', 'r') as f:
            hash_to_filename = json.load(f)
    except:
        hash_to_filename = {}
    
    # Display chunks grouped by document
    print_header("\n📄 COLLECTIONS BY DOCUMENT")
    print_separator('-')
    
    for doc_num, source in enumerate(sources, 1):
        source_hash = source['source_hash']
        chunk_count = source['chunk_count']
        filename = hash_to_filename.get(source_hash, f"Unknown Document ({source_hash[:8]}...)")
        
        print(f"\n{Colors.BOLD}{doc_num}. {filename}{Colors.ENDC}")
        print(f"   Hash: {source_hash}")
        print(f"   Chunks: {chunk_count}")
        
        # Load collection details
        coll = vdb._load_collection(source_hash)
        if not coll:
            print(f"   {Colors.FAIL}Error loading collection{Colors.ENDC}")
            continue
        
        # Group by chunk type
        by_type = defaultdict(int)
        by_page = defaultdict(int)
        
        for meta in coll['metadata']:
            by_type[meta.get('chunk_type', 'unknown')] += 1
            by_page[meta.get('page', 0)] += 1
        
        print(f"\n   {Colors.BOLD}Chunk Types:{Colors.ENDC}")
        for chunk_type, count in sorted(by_type.items()):
            print(f"     - {chunk_type}: {count}")
        
        print(f"\n   {Colors.BOLD}Pages:{Colors.ENDC}")
        for page, count in sorted(by_page.items()):
            print(f"     - Page {page}: {count} chunks")
        
        # Option to view detailed chunks
        print(f"\n   {Colors.WARNING}View detailed chunks? (y/n): {Colors.ENDC}", end='')
        choice = input().strip().lower()
        
        if choice == 'y':
            print_header(f"\n   DETAILED CHUNKS FOR: {filename}")
            
            for i, meta in enumerate(coll['metadata'], 1):
                chunk_id = meta.get('chunk_id', f'chunk_{i}')
                
                print(f"\n   {Colors.OKBLUE}Chunk {i}/{len(coll['metadata'])}{Colors.ENDC}")
                print(f"   ID: {chunk_id}")
                print(f"   Type: {meta.get('chunk_type')}")
                print(f"   Page: {meta.get('page')}")
                
                # Show FULL embedding text (now top-level metadata)
                embedding_text = meta.get('embedding_text', '')
                if embedding_text:
                    print(f"\n   {Colors.BOLD}Embedding Text:{Colors.ENDC}")
                    print(f"   {embedding_text}")  # Show full text
                
                # Show Content
                content_raw = meta.get('content', '')
                
                if meta.get('chunk_type') == 'table_generic':
                    # For tables, content is JSON string of rows
                    try:
                        rows = json.loads(content_raw)
                        print(f"\n   {Colors.BOLD}Table Info:{Colors.ENDC}")
                        print(f"   Total Rows: {len(rows)}")
                        
                        if rows:
                            print(f"\n   {Colors.BOLD}All Table Rows:{Colors.ENDC}")
                            for row_idx, row in enumerate(rows, 1):
                                row_str = ' | '.join(str(cell) for cell in row)
                                print(f"   Row {row_idx}: {row_str}")
                    except json.JSONDecodeError:
                        print(f"   {Colors.FAIL}Error decoding table rows{Colors.ENDC}")
                
                else:
                    # For non-tables, content is raw text
                    if content_raw:
                        print(f"\n   {Colors.BOLD}Content:{Colors.ENDC}")
                        print(f"   {content_raw}")  # Show full content
                
                print_separator('-', 60)
                
                # Pagination
                if i % 5 == 0 and i < len(coll['metadata']):
                    print(f"\n   {Colors.WARNING}Continue? (y/n): {Colors.ENDC}", end='')
                    if input().strip().lower() != 'y':
                        break
    
    print_separator()
    print(f"\n{Colors.OKGREEN}✓ Inspection complete!{Colors.ENDC}\n")


def search_chunks(query: str):
    """Test search functionality"""
    print_header("\n🔍 TESTING SEARCH")
    print(f"Query: {Colors.BOLD}{query}{Colors.ENDC}\n")
    
    vdb = get_vdb()
    results = vdb.query(query, n_results=5)
    
    print(f"Found {len(results)} results:\n")
    
    # Load filename mapping
    try:
        with open('faiss_store/hash_to_filename.json', 'r') as f:
            hash_to_filename = json.load(f)
    except:
        hash_to_filename = {}
    
    for i, result in enumerate(results, 1):
        source_hash = result.get('source_hash', 'unknown')
        filename = hash_to_filename.get(source_hash, 'Unknown')
        
        print(f"{Colors.OKBLUE}Result {i}:{Colors.ENDC}")
        print(f"  Document: {filename}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Type: {result['chunk_type']}")
        print(f"  Page: {result['page']}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text:\n{result['text']}")  # Show full text
        print()


def show_statistics():
    """Show quick statistics"""
    vdb = get_vdb()
    sources = vdb.get_all_sources()
    
    print(f"\n{Colors.BOLD}Quick Statistics:{Colors.ENDC}")
    print(f"  Total collections: {len(sources)}")
    print(f"  Total chunks: {sum(s['chunk_count'] for s in sources)}")
    
    # Load filename mapping
    try:
        with open('faiss_store/hash_to_filename.json', 'r') as f:
            hash_to_filename = json.load(f)
    except:
        hash_to_filename = {}
    
    print(f"\n{Colors.BOLD}Documents:{Colors.ENDC}")
    for source in sources:
        source_hash = source['source_hash']
        filename = hash_to_filename.get(source_hash, f"{source_hash[:16]}...")
        print(f"  - {filename}: {source['chunk_count']} chunks")


def main():
    """Main menu"""
    while True:
        print(f"\n{Colors.BOLD}{Colors.HEADER}=== Vector Store Inspection Tool (Collections) ==={Colors.ENDC}\n")
        print("1. Inspect all collections (grouped by document)")
        print("2. Test search query")
        print("3. Show statistics only")
        print("4. Exit")
        
        choice = input(f"\n{Colors.BOLD}Select option (1-4): {Colors.ENDC}").strip()
        
        if choice == '1':
            inspect_vector_store()
        elif choice == '2':
            query = input(f"\n{Colors.BOLD}Enter search query: {Colors.ENDC}").strip()
            if query:
                search_chunks(query)
        elif choice == '3':
            show_statistics()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print(f"{Colors.FAIL}Invalid choice{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
