from dotenv import load_dotenv
load_dotenv()

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router


app = FastAPI(title="KnowForge")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)
