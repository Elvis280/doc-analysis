"""
app.py: FastAPI backend for Campus Sathi RAG System.
Provides REST API endpoints for document management and querying.
"""

import os
import time
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Retrieval import get_vdb, get_file_hash, run_rag_pipeline
from Answer import answer_query, extract_entities_llm, build_context, generate_answer, optimize_chunks


# Initialize FastAPI app
app = FastAPI(
    title="Campus Sathi API",
    description="RAG-based Document Q&A System for Academic PDFs",
    version="1.0.0"
)

# CORS Configuration - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://campus-sathi-1.onrender.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (omitted code) ...

from fastapi import BackgroundTasks, Response

# ... (omitted code) ...

@app.get("/api/documents/status/{document_id}", tags=["Documents"])
async def get_document_status(document_id: str, response: Response):
    """
    Check the processing status of a document.
    """
    # Disable caching for this endpoint (crucial for Cloudflare Tunnels)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    status = UPLOAD_STATUS.get(document_id)
    
    # If not in memory, check if it exists in VDB (persistence check)
    if not status:
        try:
            vdb = get_vdb()
            if vdb.check_processed(document_id):
                status = "completed"
            else:
                status = "not_found"
        except:
            status = "unknown"
            
    return {"status": status, "document_id": document_id}
# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================
# Pydantic Models for Request/Response
# ============================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask about the document(s)")
    document_id: Optional[str] = Field(None, description="Specific document hash to query (optional)")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    reasoning: str  # NEW: LLM reasoning/thinking process
    entities: Dict[str, Any]
    sources: List[Dict[str, Any]]
    processing_time_ms: int

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    indexed_at: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    vector_db_size_mb: float


# ============================================
# Helper Functions
# ============================================

def get_hash_mapping() -> Dict[str, str]:
    """Load hash to filename mapping"""
    mapping_file = "faiss_store/hash_to_filename.json"
    if os.path.exists(mapping_file):
        import json
        with open(mapping_file, 'r') as f:
            return json.load(f)
    return {}

def save_hash_mapping(mapping: Dict[str, str]):
    """Save hash to filename mapping"""
    import json
    os.makedirs("faiss_store", exist_ok=True)
    with open("faiss_store/hash_to_filename.json", 'w') as f:
        json.dump(mapping, f, indent=2)

def get_pdf_path_from_hash(pdf_hash: str) -> Optional[str]:
    """Get PDF path from hash"""
    mapping = get_hash_mapping()
    filename = mapping.get(pdf_hash)
    if filename:
        safe_filename = f"{pdf_hash}.pdf"
        file_path = DATA_DIR / safe_filename
        if file_path.exists():
            return str(file_path)
    return None


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Campus Sathi RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/documents/upload",
            "list": "GET /api/documents",
            "query": "POST /api/query",
            "health": "GET /api/health",
            "stats": "GET /api/stats"
        }
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if vector DB can be initialized
        vdb = get_vdb()
        return HealthResponse(
            status="healthy",
            message="API is running and vector DB is accessible"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/api/stats", response_model=StatsResponse, tags=["Health"])
async def get_stats():
    """Get system statistics"""
    try:
        vdb = get_vdb()
        sources = vdb.get_all_sources()
        
        total_docs = len(sources)
        total_chunks = sum(s["chunk_count"] for s in sources)
        
        # Calculate vector DB size
        db_size = 0
        faiss_dir = Path("faiss_store")
        if faiss_dir.exists():
            for file in faiss_dir.glob("*"):
                if file.is_file():
                    db_size += file.stat().st_size
        
        return StatsResponse(
            total_documents=total_docs,
            total_chunks=total_chunks,
            vector_db_size_mb=round(db_size / (1024 * 1024), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# ============================================
# Global Status Store (In-Memory)
# ============================================
UPLOAD_STATUS: Dict[str, str] = {}  # document_id -> "processing" | "completed" | "failed" | "error_msg"

def process_pdf_background(file_path: Path, pdf_hash: str, original_filename: str):
    """
    Background task to process PDF: OCR -> Embeddings -> FAISS
    """
    try:
        print(f"[BACKGROUND] Starting processing for {original_filename} ({pdf_hash})")
        UPLOAD_STATUS[pdf_hash] = "processing"
        
        # heavy lifting
        run_rag_pipeline(str(file_path), query="", top_k=1)
        
        # Update hash mapping
        mapping = get_hash_mapping()
        mapping[pdf_hash] = original_filename
        save_hash_mapping(mapping)
        
        UPLOAD_STATUS[pdf_hash] = "completed"
        print(f"[BACKGROUND] Completed processing for {original_filename}")
        
    except Exception as e:
        print(f"[BACKGROUND] Failed processing for {original_filename}: {e}")
        UPLOAD_STATUS[pdf_hash] = "failed"
        # Optional: Store error message if needed, e.g., UPLOAD_STATUS[f"{pdf_hash}_error"] = str(e)
        
        # Clean up file if indexing failed
        if file_path.exists():
            file_path.unlink()

from fastapi import BackgroundTasks

@app.post("/api/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF document and start background processing.
    Returns immediately with status "processing".
    """
    print(f"[UPLOAD] Received file: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = DATA_DIR / file.filename
        
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Get file hash
        pdf_hash = get_file_hash(str(file_path))
        print(f"[UPLOAD] File hash: {pdf_hash}")
        
        # Check if already indexed
        vdb = get_vdb()
        if vdb.check_processed(pdf_hash):
            print(f"[UPLOAD] Document already indexed: {pdf_hash}")
            return UploadResponse(
                document_id=pdf_hash,
                filename=file.filename,
                status="already_indexed",
                chunks_created=0,
                message="Document was already indexed"
            )
            
        # Check if currently processing
        if UPLOAD_STATUS.get(pdf_hash) == "processing":
             return UploadResponse(
                document_id=pdf_hash,
                filename=file.filename,
                status="processing",
                chunks_created=0,
                message="Document is currently being processed"
            )
        
        # Start background task
        background_tasks.add_task(process_pdf_background, file_path, pdf_hash, file.filename)
        UPLOAD_STATUS[pdf_hash] = "processing"
        
        return UploadResponse(
            document_id=pdf_hash,
            filename=file.filename,
            status="processing",
            chunks_created=0,
            message="Document uploaded successfully. Processing started."
        )
        
    except Exception as e:
        # Clean up file if upload setup failed (before background task)
        if 'file_path' in locals() and file_path.exists() and UPLOAD_STATUS.get(pdf_hash) != "processing":
             file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to initiate upload: {str(e)}")

# Duplicate route removed here

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """
    List all indexed documents.
    """
    try:
        vdb = get_vdb()
        sources = vdb.get_all_sources()
        mapping = get_hash_mapping()
        
        documents = []
        for source in sources:
            pdf_hash = source["source_hash"]
            filename = mapping.get(pdf_hash, f"Unknown_{pdf_hash[:8]}")
            
            documents.append(DocumentInfo(
                document_id=pdf_hash,
                filename=filename,
                chunk_count=source["chunk_count"],
                indexed_at=None  # Could add timestamp to metadata if needed
            ))
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/api/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """
    Delete a document from the index and filesystem.
    """
    try:
        vdb = get_vdb()
        
        # Check if document exists
        if not vdb.check_processed(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector DB
        vdb.delete_source(document_id)
        
        # Delete file from disk
        mapping = get_hash_mapping()
        filename = mapping.get(document_id)
        if filename:
            file_path = DATA_DIR / filename
            if file_path.exists():
                file_path.unlink()
            
            # Update mapping
            del mapping[document_id]
            save_hash_mapping(mapping)
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Ask questions about indexed documents.
    """
    start_time = time.time()
    
    try:
        # Extract entities from query
        entities = extract_entities_llm(request.query)
        
        # Build filters
        filters = {}
        if request.document_id:
            # Verify document exists
            vdb = get_vdb()
            if not vdb.check_processed(request.document_id):
                raise HTTPException(status_code=404, detail="Document not found")
            filters["source_hash"] = request.document_id
        
        # NOTE: Intent filtering removed since we removed intent from chunks
        # if "_intent" in entities:
        #     filters["intent"] = entities["_intent"]
        
        # Retrieve more chunks initially for better optimization
        vdb = get_vdb()
        initial_top_k = min(request.top_k * 4, 20)
        results = vdb.query(request.query, n_results=initial_top_k, where_filter=filters)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the indexed documents"
            )
        
        # *** OPTIMIZATION PIPELINE ***
        results = optimize_chunks(
            chunks=results,
            entities=entities,
            max_tokens=2500,
            max_per_page=3
        )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found after optimization"
            )
        
        # Generate reasoning (NEW)
        from Answer import generate_reasoning
        reasoning = generate_reasoning(request.query, entities, results)
        
        # Build context with row filtering
        context = build_context(results, entities=entities)
        
        if not context.strip():
            raise HTTPException(
                status_code=404,
                detail="No relevant information found after filtering"
            )
        
        # Generate answer
        answer = generate_answer(request.query, context)
        
        # Prepare source information
        sources = []
        mapping = get_hash_mapping()
        for result in results[:5]:
            sources.append({
                "page": result.get("page", "N/A"),
                "chunk_type": result.get("chunk_type", "unknown"),
                "document": mapping.get(result.get("source_hash", ""), "Unknown"),
                "relevance_score": round(result.get("distance", 0), 3)
            })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=answer,
            reasoning=reasoning,  # NEW
            entities=entities,
            sources=sources,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Campus Sathi API Server")
    print("=" * 60)
    print("Starting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
