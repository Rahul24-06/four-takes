"""FOUR TAKES — FastAPI server with a persistent run library."""

import base64
import datetime
import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import run_pipeline

app = FastAPI(title="FOUR TAKES")
JOBS: dict = {}  # job_id -> {status, stage, events[], result}

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
CLIPS = DATA / "clips"
RUNS = DATA / "runs"
for d in (DATA, CLIPS, RUNS):
    d.mkdir(exist_ok=True)

MEDIA = {".mp4": "video/mp4", ".mov": "video/quicktime",
         ".webm": "video/webm", ".mkv": "video/x-matroska",
         ".avi": "video/x-msvideo"}


def _thumb_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _worker(job_id: str, video_path: Path, clip_name: str, n_frames):
    job = JOBS[job_id]

    def progress(stage, data=None):
        data = data or {}
        if stage == "sampled" and "frames" in data:
            data = {"frame_count": data["frame_count"],
                    "times": data.get("times", []),
                    "thumbs": [_thumb_b64(f) for f in data["frames"][:32]]}
        job["stage"] = stage
        job["events"].append({"stage": stage, "data": data})

    try:
        result = run_pipeline(str(video_path), progress, n_frames=n_frames)
        job["result"] = result
        job["status"] = "done"
        # persist the run: captions + metadata; clip already lives in CLIPS
        (RUNS / f"{job_id}.json").write_text(json.dumps({
            "id": job_id,
            "clip_name": clip_name,
            "clip_file": video_path.name,
            "ts": datetime.datetime.utcnow().isoformat(),
            "result": result,
        }, indent=1))
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        video_path.unlink(missing_ok=True)  # keep storage clean on failures


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), frames: int = Form(0)):
    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    job_id = uuid.uuid4().hex[:12]
    clip_path = CLIPS / f"{job_id}{suffix}"
    with open(clip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[job_id] = {"status": "running", "stage": "queued",
                    "events": [], "result": None}
    threading.Thread(target=_worker,
                     args=(job_id, clip_path, file.filename or clip_path.name,
                           frames or None),
                     daemon=True).start()
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


# ------------------------------------------------------------- library ----

@app.get("/api/library")
async def library():
    items = []
    for f in sorted(RUNS.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
            items.append({
                "id": d["id"], "clip_name": d["clip_name"], "ts": d["ts"],
                "captions": {k: v["caption"]
                             for k, v in d["result"]["captions"].items()},
            })
        except Exception:
            continue
    return {"items": items}


@app.get("/api/library/{rid}")
async def library_item(rid: str):
    f = RUNS / f"{rid}.json"
    if not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(f.read_text())


@app.get("/api/library/{rid}/clip")
async def library_clip(rid: str):
    for f in CLIPS.glob(f"{rid}.*"):
        return FileResponse(f, media_type=MEDIA.get(f.suffix, "video/mp4"))
    return JSONResponse({"error": "clip not found"}, status_code=404)


@app.delete("/api/library/{rid}")
async def library_delete(rid: str):
    (RUNS / f"{rid}.json").unlink(missing_ok=True)
    for f in CLIPS.glob(f"{rid}.*"):
        f.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/waitlist")
async def waitlist(payload: dict):
    email = str(payload.get("email", "")).strip()
    plan = str(payload.get("plan", "waitlist")).strip()
    if "@" not in email or "." not in email or len(email) > 120:
        return JSONResponse({"error": "invalid email"}, status_code=400)
    with open(DATA / "waitlist.jsonl", "a") as fh:
        fh.write(json.dumps({"email": email, "plan": plan,
                             "ts": datetime.datetime.utcnow().isoformat()})
                 + "\n")
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"))
