"""FOUR TAKES — FastAPI server. Serves the screening-room UI and the pipeline."""

import base64
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import run_pipeline

app = FastAPI(title="FOUR TAKES")
JOBS: dict = {}  # job_id -> {status, stage, events[], result}


def _thumb_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _worker(job_id: str, video_path: str):
    job = JOBS[job_id]

    def progress(stage, data=None):
        data = data or {}
        if stage == "sampled" and "frames" in data:
            data = {"frame_count": data["frame_count"],
                    "thumbs": [_thumb_b64(f) for f in data["frames"][:8]]}
        job["stage"] = stage
        job["events"].append({"stage": stage, "data": data})

    try:
        job["result"] = run_pipeline(video_path, progress)
        job["status"] = "done"
    except Exception as e:  # surface real errors to the UI
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        Path(video_path).unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(file.file, f)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "stage": "queued", "events": [], "result": None}
    threading.Thread(target=_worker, args=(job_id, tmp.name), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, cursor: int = 0):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"status": job["status"], "stage": job["stage"],
            "events": job["events"][cursor:], "cursor": len(job["events"]),
            "result": job["result"] if job["status"] == "done" else None,
            "error": job.get("error")}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"))
