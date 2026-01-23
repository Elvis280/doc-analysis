# Backend - Campus Sathi

FastAPI-powered backend with RAG (Retrieval-Augmented Generation) capabilities for intelligent document querying.

## 🏗️ Architecture

```
PDF Upload → Extract (OCR) → Clean (Chunk) → Embed → VectorStore (FAISS)
                                                           ↓
User Query → Retrieval (Semantic Search) → Answer (LLM) → Response
```

## 📦 Tech Stack

- **Framework**: FastAPI (async)
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: Nomic AI (nomic-embed-text-v1)
- **Vector DB**: FAISS
- **OCR**: PaddleOCR
- **Text Processing**: LangChain

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

3. **Run server**
```bash
python app.py
```

Server starts at `http://localhost:8000`  
API docs at `http://localhost:8000/docs`

## 📂 File Structure

```
Backend/
├── app.py              # FastAPI server & endpoints
├── Answer.py           # LLM answer generation with reasoning
├── Retrieval.py        # Query pipeline & vector search
├── Extract.py          # PDF OCR & text extraction
├── Clean.py            # Text chunking & classification
├── VectorStore.py      # FAISS vector database management
├── Embeddings.py       # Nomic embedding wrapper
├── Normal.py           # Text normalization utilities
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (gitignored)
├── data/              # Uploaded PDFs (gitignored)
├── faiss_store/       # Vector indices (gitignored)
└── tests/             # Test scripts
```

## 🔌 API Endpoints

### Query Documents
```http
POST /api/query
Content-Type: application/json

{
  "query": "When is the DBMS exam?",
  "document_id": "optional_doc_id",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The DBMS exam is on...",
  "reasoning": "Based on the query...",
  "sources": [...],
  "entities": {...},
  "processing_time_ms": 1234
}
```

### Upload Document
```http
POST /api/documents/upload
Content-Type: multipart/form-data

file: <PDF file>
```

### List Documents
```http
GET /api/documents
```

### Delete Document
```http
DELETE /api/documents/{document_id}
```

### Health Check
```http
GET /api/health
```

### System Stats
```http
GET /api/stats
```

## ⚙️ Configuration

**Environment Variables** (`.env`):
```env
GROQ_API_KEY=gsk_...
```

**Default Settings** (configurable in code):
- `MAX_TOKENS`: 2500 (context window)
- `CHUNK_SIZE`: 1000 characters
- `CHUNK_OVERLAP`: 200 characters
- `TOP_K`: 5 results
- `LLM_TEMPERATURE`: 0.2
- `LLM_MAX_TOKENS`: 512

## 🧠 Key Components

### 1. Extract.py
- PDF text extraction using PaddleOCR
- Table detection and cell-level OCR
- Page-level processing
- Handles merged cells and complex layouts

### 2. Clean.py
- Semantic chunking with RecursiveCharacterTextSplitter
- Automatic chunk classification (rule, notice, instruction, etc.)
- Exam slot grouping for exam schedules
- Section-based organization

### 3. VectorStore.py
- Per-document FAISS collections
- Persistent storage in `faiss_store/`
- Efficient similarity search
- Metadata tracking

### 4. Retrieval.py
- Query embedding with Nomic
- Vector similarity search
- Top-k ranking
- Context building

### 5. Answer.py
- Entity extraction (roll numbers, dates, subjects)
- Range-aware filtering for tables
- LLM prompt engineering
- Reasoning generation

## 🧪 Testing

```bash
cd tests

# Inspect vector store chunks
python test_chunks.py

# Re-index all PDFs (clears and rebuilds)
python reindex.py
```

## 📊 Performance

- **Embedding**: ~50ms per chunk (batch processing)
- **Search**: ~10ms for top-5 results
- **LLM**: ~500-1500ms (depends on Groq load)
- **Total Query**: ~600-2000ms end-to-end

## 🛠️ Development

### Adding a New Endpoint

1. Define request/response models in `app.py`
2. Create endpoint function with `@app.post()` or `@app.get()`
3. Add CORS if needed
4. Test with `/docs` Swagger UI

### Modifying Chunking Logic

Edit `Clean.py`:
- Update `chunk_text()` for new chunk sizes
- Modify classification rules in `classify_chunk_type()`
- Adjust exam slot grouping in `group_exam_slots()`

### Changing LLM or Embeddings

1. **LLM**: Update `Answer.py` → `generate_answer_with_reasoning()`
2. **Embeddings**: Update `Embeddings.py` → `__init__()` model name

## 🐛 Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt --force-reinstall
```

**GROQ_API_KEY not found?**
- Ensure `.env` file exists in `Backend/`
- Check variable name is exactly `GROQ_API_KEY`

**OCR not working?**
- PaddleOCR downloads models on first run (~100MB)
- Check internet connection
- Verify disk space for model cache

**Vector store corrupted?**
```bash
cd tests
python reindex.py  # Rebuilds all indices
```

## 📝 Notes

- All PDFs in `data/` are automatically indexed on startup
- Vector indices cached in `faiss_store/collections/`
- Each PDF gets its own FAISS collection
- Document IDs are MD5 hashes of file content

## 🔐 Security

- API key stored in `.env` (gitignored)
- No authentication on endpoints (add auth middleware if needed)
- CORS enabled for frontend (localhost:5173, localhost:8080)
- File uploads limited to PDF only

---

For more details, see the [main README](../README.md) or [API documentation](http://localhost:8000/docs).
