import fitz  # PyMuPDF
from typing import List, Callable, Optional
from app.schemas import KnowledgeUnit
from app.llm import extract_knowledge_units


async def process_pdf(
    file_path: str,
    document_name: str,
    log: Optional[Callable[[str], None]] = None,
) -> List[KnowledgeUnit]:
    """
    Extracts text from PDF, splits into page chunks, and extracts knowledge units.
    Fully async — each Gemini call is awaited without blocking the event loop.

    Args:
        log: Optional callback invoked with each progress message so callers
             (e.g. the job runner in routes.py) can surface them to the UI.
    """
    def _log(msg: str):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))
        if log:
            log(msg)

    doc = fitz.open(file_path)
    try:
        all_units: List[KnowledgeUnit] = []

        PAGES_PER_CHUNK = 1
        total_pages = len(doc)
        total_chunks = (total_pages + PAGES_PER_CHUNK - 1) // PAGES_PER_CHUNK

        _log(f"📄 Opened '{document_name}' — {total_pages} page(s), {total_chunks} chunk(s) to process.")

        for chunk_idx, i in enumerate(range(0, total_pages, PAGES_PER_CHUNK), start=1):
            start_page = i
            end_page = min(i + PAGES_PER_CHUNK - 1, total_pages - 1)

            chunk_text = ""
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                chunk_text += f"\n\n--- Page {page_num + 1} ---\n\n"
                chunk_text += page.get_text("text")

            if not chunk_text.strip():
                _log(f"⚠️  Chunk {chunk_idx}/{total_chunks} (pages {start_page + 1}–{end_page + 1}) appears empty — skipping.")
                continue

            _log(f"🔍 Chunk {chunk_idx}/{total_chunks} — extracting knowledge units from pages {start_page + 1}–{end_page + 1}…")

            units = await extract_knowledge_units(
                chunk_text, document_name, start_page + 1, end_page + 1
            )
            all_units.extend(units)
            _log(f"✅ Chunk {chunk_idx}/{total_chunks} — extracted {len(units)} unit(s). Running total: {len(all_units)}.")

        _log(f"🎉 Finished. Total knowledge units extracted: {len(all_units)}.")
        return all_units
    finally:
        doc.close()
