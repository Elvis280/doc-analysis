# RAG Knowledge Base Generator

A full-stack, highly optimized asynchronous tool for extracting structured knowledge units from PDF documents. This generator is perfect for building high-quality, dense context sources for Retrieval-Augmented Generation (RAG) pipelines.

## Features

- **Asynchronous Processing Pipeline**: Built on FastAPI and `asyncio`, the parsing and extraction process works asynchronously in the background so the event loop is never blocked.
- **Advanced Knowledge Extraction**: Uses the Groq API (leveraging high-parameter models) to parse text chunks and extract semantically meaningful "knowledge units" rather than arbitrary chunks.
- **Resilient AI Calling**: Uses `tenacity` for exponential backoff and retry handling (resilient to 429 Rate Limits and 5xx server errors).
- **Beautiful UI**: Modern glassmorphism frontend built in vanilla HTML/CSS/JS with real-time SSE-like polling for job status and progress tracking.
- **Export to ZIP**: Automatically compiles the extracted units into neat Markdown files and structured JSON metadata inside a downloadable ZIP file.
- **Privacy & Cleanup**: Strictly ephemeral. Uploaded files and generated archives are automatically wiped from the server as soon as the download finishes.

## Prerequisites

- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)

## Setup & Installation

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <your-repo-url>
   cd rag-kb-generator
   ```

2. **Set up a virtual environment** (recommended with `uv` or `venv`):
   ```bash
   uv venv
   # Or standard python: python -m venv .venv
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r pyproject.toml
   # Or run: pip install fastapi uvicorn pydantic pymupdf groq tenacity python-dotenv jinja2 python-multipart
   ```

4. **Environment Variables**:
   Create a `.env` file in the project root and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

## Running the Application

Start the FastAPI backend with `uvicorn`:
```bash
uv run uvicorn app.main:app --reload
```
Then, open your web browser and navigate to: `http://127.0.0.1:8000/`

## Project Structure

- `app/main.py`: Application entry point and server configuration.
- `app/routes.py`: FastAPI routes, HTTP handlers, background tasks, and cleanup.
- `app/pipeline.py`: Asynchronous document processing and chunking logic.
- `app/llm.py`: Interaction with the Groq LLM and extraction prompting.
- `app/exporter.py`: File formatting (Markdown & JSON) and ZIP compilation.
- `app/schemas.py`: Pydantic models for type safety across the application.
- `templates/`: HTML views for the frontend UI.
- `static/`: Custom CSS and frontend assets.

## License

MIT License.
