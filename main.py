"""
main.py: Entry point for the RAG Document Q&A System.
Handles document scanning, indexing, and the interactive Q&A loop.
"""

import os
import glob
import hashlib
from typing import List, Dict, Tuple
from Retrieval import get_vdb, get_file_hash, run_rag_pipeline
from Answer import answer_query, extract_entities_llm

# Color codes for terminal (optional, works on most terminals)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def scan_data_folder(data_path: str = "data") -> List[str]:
    """
    Scans data folder for all PDF files.
    Returns list of absolute paths.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print_warning(f"Created {data_path}/ folder. Please add PDF files there.")
        return []
    
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    return [os.path.abspath(f) for f in pdf_files]


def index_all_pdfs(pdf_paths: List[str]) -> Dict[str, str]:
    """
    Indexes all PDFs that aren't already indexed.
    Returns dict mapping pdf_hash -> pdf_path
    """
    vdb = get_vdb()
    hash_to_path = {}
    
    new_count = 0
    
    for pdf_path in pdf_paths:
        pdf_hash = get_file_hash(pdf_path)
        hash_to_path[pdf_hash] = pdf_path
        
        if not vdb.check_processed(pdf_hash):
            print_info(f"Indexing: {os.path.basename(pdf_path)}...")
            try:
                # Run pipeline without query to just index
                run_rag_pipeline(pdf_path, query="", top_k=1)
                print_success(f"Indexed: {os.path.basename(pdf_path)}")
                new_count += 1
            except Exception as e:
                print_warning(f"Failed to index {os.path.basename(pdf_path)}: {e}")
        else:
            print_success(f"Already indexed: {os.path.basename(pdf_path)}")
    
    if new_count > 0:
        print_success(f"\nIndexed {new_count} new document(s)!")
    
    return hash_to_path


def get_indexed_documents() -> List[Tuple[str, str, int]]:
    """
    Returns list of (pdf_hash, pdf_name, chunk_count) for all indexed docs.
    Note: We need to maintain a mapping file for hash -> filename
    """
    vdb = get_vdb()
    sources = vdb.get_all_sources()
    
    # Load hash-to-filename mapping
    mapping_file = "faiss_store/hash_to_filename.json"
    hash_to_filename = {}
    
    if os.path.exists(mapping_file):
        import json
        with open(mapping_file, 'r') as f:
            hash_to_filename = json.load(f)
    
    documents = []
    for source in sources:
        pdf_hash = source["source_hash"]
        chunk_count = source["chunk_count"]
        filename = hash_to_filename.get(pdf_hash, f"Unknown ({pdf_hash[:8]})")
        documents.append((pdf_hash, filename, chunk_count))
    
    return documents


def save_hash_mapping(hash_to_path: Dict[str, str]):
    """Save hash to filename mapping for future reference"""
    import json
    mapping = {h: os.path.basename(p) for h, p in hash_to_path.items()}
    
    os.makedirs("faiss_store", exist_ok=True)
    with open("faiss_store/hash_to_filename.json", 'w') as f:
        json.dump(mapping, f, indent=2)


def select_document(documents: List[Tuple[str, str, int]]) -> Tuple[str, str]:
    """
    Interactive document selection.
    Returns (pdf_hash, filename)
    """
    print_header("Available Documents")
    
    if not documents:
        print_warning("No documents indexed yet!")
        return None, None
    
    for i, (pdf_hash, filename, chunk_count) in enumerate(documents, 1):
        print(f"{Colors.BOLD}{i}.{Colors.ENDC} {filename} ({chunk_count} chunks)")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select document (1-{len(documents)}) or 'q' to quit: {Colors.ENDC}")
            
            if choice.lower() == 'q':
                return None, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(documents):
                pdf_hash, filename, _ = documents[idx]
                return pdf_hash, filename
            else:
                print_warning(f"Please enter a number between 1 and {len(documents)}")
        except ValueError:
            print_warning("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\n")
            return None, None


def qa_loop(pdf_hash: str, filename: str):
    """
    Interactive Q&A loop for selected document.
    """
    print_header(f"Q&A: {filename}")
    print_info("Ask questions about this document")
    print_info("Type 'back' to select another document, or 'quit' to exit\n")
    
    while True:
        try:
            question = input(f"{Colors.BOLD}Question: {Colors.ENDC}").strip()
            
            if not question:
                continue
            
            if question.lower() in ['back', 'b']:
                return 'back'
            
            if question.lower() in ['quit', 'q', 'exit']:
                return 'quit'
            
            print(f"\n{Colors.OKBLUE}Thinking...{Colors.ENDC}")
            
            # Extract entities for filtering
            entities = extract_entities_llm(question)
            
            # Build filters
            filters = {"source_hash": pdf_hash}  # Always filter by selected doc
            if "_intent" in entities:
                filters["intent"] = entities["_intent"]
            
            # Query the system
            vdb = get_vdb()
            results = vdb.query(question, n_results=5, where_filter=filters)
            
            if not results:
                print(f"\n{Colors.WARNING}No relevant information found in this document.{Colors.ENDC}\n")
                continue
            
            # Build context and generate answer (using Answer.py logic)
            from Answer import build_context, generate_answer
            context = build_context(results, entities=entities)
            
            if not context.strip():
                print(f"\n{Colors.WARNING}No relevant information found (filtered out).{Colors.ENDC}\n")
                continue
            
            answer = generate_answer(question, context)
            
            print(f"\n{Colors.OKGREEN}Answer:{Colors.ENDC}")
            print(f"{answer}\n")
            
        except KeyboardInterrupt:
            print("\n")
            return 'back'
        except Exception as e:
            print_warning(f"Error: {e}\n")


def main():
    """Main interactive loop"""
    print_header("RAG Document Q&A System")
    
    while True:
        # 1. Scan for PDFs
        print_info("Scanning data/ folder for PDFs...")
        pdf_paths = scan_data_folder("data")
        
        if not pdf_paths:
            print_warning("No PDF files found in data/ folder!")
            print("Please add PDF files to the data/ folder and restart.")
            break
        
        print_success(f"Found {len(pdf_paths)} PDF(s)\n")
        
        # 2. Index new PDFs
        print_header("Indexing Documents")
        hash_to_path = index_all_pdfs(pdf_paths)
        save_hash_mapping(hash_to_path)
        
        # 3. Get indexed documents
        documents = get_indexed_documents()
        
        if not documents:
            print_warning("No documents indexed. Please check for errors above.")
            break
        
        # 4. Document selection
        pdf_hash, filename = select_document(documents)
        
        if pdf_hash is None:
            print("\nGoodbye!")
            break
        
        # 5. Q&A Loop
        result = qa_loop(pdf_hash, filename)
        
        if result == 'quit':
            print("\nGoodbye!")
            break
        # If 'back', loop continues to document selection


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
