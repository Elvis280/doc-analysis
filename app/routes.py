import os
import shutil
import uuid
from datetime import datetime, date
from typing import Dict

import fitz  # PyMuPDF
import asyncio
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from starlette.background import BackgroundTask

from app.pipeline import process_pdf
from app.exporter import export_kb
from app.schemas import JobInfo, JobStatus
from app.llm import API_STATUS

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
OUTPUT_BASE_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
_jobs: Dict[str, JobInfo] = {}

GLOBAL_USAGE = {
    "date": date.today().isoformat(),
    "docs_processed": 0,
    "daily_limit": 5
}

def cleanup_job_data(job_id: str, job_output_dir: str):
    """Deletes job info and all generated files to ensure no data is stored permanently."""
    if job_id in _jobs:
        del _jobs[job_id]
    if os.path.exists(job_output_dir):
        shutil.rmtree(job_output_dir, ignore_errors=True)

async def auto_delete_job(job_id: str, job_output_dir: str, delay_seconds: int = 900):
    """Fallback cleanup: deletes the job after 15 minutes if abandoned."""
    await asyncio.sleep(delay_seconds)
    cleanup_job_data(job_id, job_output_dir)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
async def _run_pipeline(job_id: str, file_path: str, filename: str, job_output_dir: str):
    """Async background task: runs the full PDF → KB pipeline for one job."""
    job = _jobs[job_id]
    job.status = JobStatus.processing
    job.updated_at = datetime.utcnow()

    def log(msg: str):
        """Append a timestamped log message to the job."""
        ts = datetime.utcnow().strftime("%H:%M:%S")
        job.logs.append(f"[{ts}] {msg}")
        job.updated_at = datetime.utcnow()

    log(f"🚀 Job started for '{filename}'.")

    try:
        units = await process_pdf(file_path, filename, log=log)

        if not units:
            raise ValueError("No knowledge units could be generated from the document.")

        log(f"📦 Exporting {len(units)} unit(s) to ZIP…")
        zip_path = export_kb(units, filename, job_output_dir)

        job.status = JobStatus.done
        job.num_units = len(units)
        job.zip_path = zip_path
        log(f"✅ Export complete. Ready to download.")
    except Exception as exc:
        job.status = JobStatus.failed
        job.error = str(exc)
        log(f"❌ Job failed: {exc}")
        log("⚠️ Please start a new job to try again.")
        
        # Refund the global counter so they don't lose their attempt
        if GLOBAL_USAGE["docs_processed"] > 0:
            GLOBAL_USAGE["docs_processed"] -= 1

        # Clean up residue files immediately, but keep the job in memory 
        # so the frontend can still fetch the logs and display the error.
        if os.path.exists(job_output_dir):
            shutil.rmtree(job_output_dir, ignore_errors=True)
    finally:
        job.updated_at = datetime.utcnow()
        # Clean up the uploaded PDF
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@router.get("/app", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@router.get("/api-limits")
async def get_api_limits():
    """Returns the global daily usage stats for the app."""
    today = date.today().isoformat()
    if GLOBAL_USAGE["date"] != today:
        GLOBAL_USAGE["date"] = today
        GLOBAL_USAGE["docs_processed"] = 0

    return {
        "docs_processed": GLOBAL_USAGE["docs_processed"],
        "daily_limit": GLOBAL_USAGE["daily_limit"],
        "remaining": max(0, GLOBAL_USAGE["daily_limit"] - GLOBAL_USAGE["docs_processed"])
    }

@router.post("/generate", status_code=202)
async def generate_kb(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF upload, enqueues a background processing job, and immediately
    returns a job_id the client can poll via GET /status/{job_id}.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    today = date.today().isoformat()
    if GLOBAL_USAGE["date"] != today:
        GLOBAL_USAGE["date"] = today
        GLOBAL_USAGE["docs_processed"] = 0

    if GLOBAL_USAGE["docs_processed"] >= GLOBAL_USAGE["daily_limit"]:
        raise HTTPException(status_code=429, detail="Global daily limit of 5 documents has been reached. Please try again tomorrow.")

    # Increment counter immediately
    GLOBAL_USAGE["docs_processed"] += 1

    # Create a unique job directory so concurrent jobs never clobber each other
    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_BASE_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Persist the upload inside the job-scoped directory
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Enforce maximum 8 page limit
    try:
        with fitz.open(file_path) as doc:
            page_count = len(doc)
            
        if page_count > 8:
            try:
                os.remove(file_path)
            except OSError:
                pass
            shutil.rmtree(job_output_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400, 
                detail=f"PDF is too large ({page_count} pages). Please limit your document to a maximum of 8 pages."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        try:
            os.remove(file_path)
        except OSError:
            pass
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")

    # Register job
    _jobs[job_id] = JobInfo(job_id=job_id, filename=file.filename)

    # Kick off pipeline in the background — response returns immediately
    background_tasks.add_task(_run_pipeline, job_id, file_path, file.filename, job_output_dir)

    # Schedule a fallback auto-deletion after 15 minutes for strict privacy
    asyncio.create_task(auto_delete_job(job_id, job_output_dir))

    return {"job_id": job_id, "status": JobStatus.pending}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll this endpoint to track job progress.
    Returns the current JobInfo including status, num_units, and any error.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("/download/{job_id}")
async def download_kb(job_id: str):
    """
    Download the generated ZIP once the job status is 'done'.
    Returns 404 if not found, 409 if not yet complete, 500 if the job failed.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == JobStatus.failed:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not yet complete (current status: {job.status}).",
        )

    if not job.zip_path or not os.path.exists(job.zip_path):
        raise HTTPException(status_code=500, detail="Output file not found on server.")

    # Clean up immediately after the user downloads it
    job_output_dir = os.path.join(OUTPUT_BASE_DIR, job_id)
    
    async def delayed_cleanup():
        await asyncio.sleep(1)
        cleanup_job_data(job_id, job_output_dir)

    cleanup_task = BackgroundTask(delayed_cleanup)

    return FileResponse(
        path=job.zip_path,
        filename="usern.zip",
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=usern.zip"},
        background=cleanup_task
    )
