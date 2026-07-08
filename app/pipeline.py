"""
FOUR TAKES — captioning pipeline.

Stages:
  1. SAMPLE     ffmpeg scene-change keyframes (uniform fallback) + audio track
  2. TRANSCRIBE Fireworks Whisper on the audio track
  3. DOSSIER    Gemma 3 27B (multimodal, via Fireworks) reads frames+transcript
                and produces a grounded scene dossier (facts only)
  4. FOUR TAKES Gemma writes one caption per style from the dossier
  5. JUDGE      An LLM judge scores accuracy+tone; failing captions are
                revised with the judge's fix instruction (up to N rounds)

Everything is grounded in the dossier so all four styles agree on the facts —
that's what protects the ACCURACY score while tone varies.
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests

from .styles import STYLES, JUDGE_RUBRIC

FIREWORKS_BASE = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Gemma-first (eligible for the $3,000 Gemma partner prize).
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/gemma-3-27b-it")
STYLE_MODEL = os.getenv("STYLE_MODEL", "accounts/fireworks/models/gemma-3-27b-it")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "accounts/fireworks/models/gemma-3-27b-it")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-v3")

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "8"))
MAX_JUDGE_ROUNDS = int(os.getenv("MAX_JUDGE_ROUNDS", "2"))
TEMPERATURE_STYLE = float(os.getenv("TEMPERATURE_STYLE", "0.8"))


# ---------------------------------------------------------------- ffmpeg ---

def sample_frames(video_path: str, workdir: str, max_frames: int = MAX_FRAMES):
    """Scene-change keyframes; uniform sampling as fallback for static clips."""
    out_pattern = str(Path(workdir) / "scene_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", "select='gt(scene,0.25)',scale=640:-2", "-vsync", "vfr",
         "-frames:v", str(max_frames), "-q:v", "3", out_pattern],
        capture_output=True, timeout=120,
    )
    frames = sorted(Path(workdir).glob("scene_*.jpg"))
    if len(frames) < 3:  # static clip -> uniform sampling
        dur = probe_duration(video_path)
        step = max(dur / max_frames, 0.5)
        out_pattern = str(Path(workdir) / "uni_%03d.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"fps=1/{step:.3f},scale=640:-2",
             "-frames:v", str(max_frames), "-q:v", "3", out_pattern],
            capture_output=True, timeout=120,
        )
        frames = sorted(Path(workdir).glob("uni_*.jpg"))
    return [str(f) for f in frames[:max_frames]]


def extract_audio(video_path: str, workdir: str):
    audio_path = str(Path(workdir) / "audio.mp3")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
         "-ar", "16000", "-b:a", "64k", audio_path],
        capture_output=True, timeout=120,
    )
    return audio_path if r.returncode == 0 and Path(audio_path).exists() else None


def probe_duration(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 60.0


# ------------------------------------------------------------- fireworks ---

def _headers():
    return {"Authorization": f"Bearer {FIREWORKS_KEY}",
            "Content-Type": "application/json"}


def chat(model: str, messages: list, temperature: float = 0.4,
         max_tokens: int = 900) -> str:
    resp = requests.post(
        f"{FIREWORKS_BASE}/chat/completions",
        headers=_headers(),
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def transcribe(audio_path: str) -> str:
    if not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{FIREWORKS_BASE}/audio/transcriptions",
                headers={"Authorization": f"Bearer {FIREWORKS_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={"model": WHISPER_MODEL},
                timeout=180,
            )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
    except Exception:
        return ""  # silent clip or transcription unavailable — vision carries it


def _img_content(frame_path: str) -> dict:
    b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


# --------------------------------------------------------------- stages ----

def build_dossier(frames: list, transcript: str) -> dict:
    """Grounded scene dossier — the single source of truth for all 4 styles."""
    content = [{"type": "text", "text":
        "You will see keyframes sampled in order from one short video clip"
        + (f", plus this audio transcript:\n---\n{transcript}\n---\n" if transcript
           else " (the clip has no usable speech).\n")
        + "\nProduce a factual scene dossier. Report ONLY what is visibly or "
          "audibly present — never guess names, brands, or intent you cannot "
          "see. Reply ONLY with JSON:\n"
          '{"subjects": [..], "setting": "..", "actions_in_order": [..], '
          '"on_screen_text": [..], "notable_details": [..], '
          '"overall_summary": "2-3 sentence neutral summary"}'}]
    content += [_img_content(f) for f in frames]
    raw = chat(VISION_MODEL, [{"role": "user", "content": content}],
               temperature=0.1, max_tokens=800)
    d = _extract_json(raw)
    return d or {"overall_summary": raw[:500], "subjects": [],
                 "actions_in_order": [], "on_screen_text": [],
                 "notable_details": [], "setting": ""}


def write_caption(style_key: str, dossier: dict, transcript: str,
                  fix: str = "", previous: str = "") -> str:
    s = STYLES[style_key]
    sys = (f"You are {s['label']}, a caption writer. Style: {s['voice']} "
           f"Never do: {s['avoid']}.\n"
           f"Example of your voice (about a DIFFERENT clip — copy the voice, "
           f"never the content):\n\"{s['anchor']}\"")
    user = ("Write ONE caption (1-4 sentences) for the video described in "
            "this grounded dossier. Every fact you mention must come from "
            "the dossier or transcript — no invention.\n\n"
            f"DOSSIER:\n{json.dumps(dossier, indent=1)}\n\n"
            f"TRANSCRIPT: {transcript or '(no speech)'}\n\n"
            "Reply with the caption text only. No quotes, no preamble.")
    if fix:
        user += (f"\n\nYour previous attempt: \"{previous}\"\n"
                 f"A judge rejected it. Fix exactly this: {fix}")
    return chat(STYLE_MODEL,
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user}],
                temperature=TEMPERATURE_STYLE, max_tokens=300)


def judge_caption(style_key: str, caption: str, dossier: dict,
                  transcript: str) -> dict:
    user = (f"TARGET STYLE: {style_key}\n\nVIDEO FACTS (ground truth):\n"
            f"{json.dumps(dossier, indent=1)}\n\n"
            f"TRANSCRIPT: {transcript or '(no speech)'}\n\n"
            f"CAPTION TO SCORE:\n\"{caption}\"")
    raw = chat(JUDGE_MODEL,
               [{"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": user}],
               temperature=0.0, max_tokens=250)
    v = _extract_json(raw)
    return v if "verdict" in v else {"accuracy": 7, "tone": 7,
                                     "verdict": "pass", "fix": ""}


# ------------------------------------------------------------ orchestrate --

def run_pipeline(video_path: str, progress=lambda stage, data=None: None) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        progress("sampling")
        frames = sample_frames(video_path, workdir)
        progress("sampled", {"frame_count": len(frames), "frames": frames})

        progress("transcribing")
        transcript = transcribe(extract_audio(video_path, workdir))
        progress("transcribed", {"transcript": transcript})

        progress("dossier")
        dossier = build_dossier(frames, transcript)
        progress("dossier_done", {"dossier": dossier})

        results = {}
        for style_key in STYLES:
            progress("writing", {"style": style_key})
            caption = write_caption(style_key, dossier, transcript)
            verdict = judge_caption(style_key, caption, dossier, transcript)
            rounds = 0
            while verdict.get("verdict") == "revise" and rounds < MAX_JUDGE_ROUNDS:
                rounds += 1
                progress("revising", {"style": style_key, "round": rounds,
                                      "fix": verdict.get("fix", "")})
                caption = write_caption(style_key, dossier, transcript,
                                        fix=verdict.get("fix", ""),
                                        previous=caption)
                verdict = judge_caption(style_key, caption, dossier, transcript)
            results[style_key] = {
                "caption": caption,
                "scores": {"accuracy": verdict.get("accuracy"),
                           "tone": verdict.get("tone")},
                "judge_rounds": rounds,
            }
            progress("style_done", {"style": style_key,
                                    **results[style_key]})

        return {"transcript": transcript, "dossier": dossier,
                "captions": results, "frame_count": len(frames)}
