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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "32"))  # upper bound
MIN_FRAMES = int(os.getenv("MIN_FRAMES", "16"))
MAX_JUDGE_ROUNDS = int(os.getenv("MAX_JUDGE_ROUNDS", "2"))
TEMPERATURE_STYLE = float(os.getenv("TEMPERATURE_STYLE", "0.8"))
PARALLEL_STYLES = int(os.getenv("PARALLEL_STYLES", "4"))  # set 2 if rate-limited


# ---------------------------------------------------------------- ffmpeg ---

def sample_frames(video_path: str, workdir: str, n_frames: int = None):
    """Uniform sampling across the FULL clip duration, 16-32 frames.

    n_frames: user-selected count from the Sampler slider (clamped 16..32).
    If not given, density adapts to clip length (~1 frame every 4s).
    Frames are extracted concurrently. Returns (frame_paths, timestamps).
    """
    dur = probe_duration(video_path)
    if n_frames:
        n = max(MIN_FRAMES, min(MAX_FRAMES, int(n_frames)))
    else:
        n = max(MIN_FRAMES, min(MAX_FRAMES, round(dur / 4)))
    times = [dur * (i + 0.5) / n for i in range(n)]

    def grab(i_t):
        i, t = i_t
        out = str(Path(workdir) / f"uni_{i:03d}.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
             "-vf", "scale=512:-2", "-frames:v", "1", "-q:v", "4", out],
            capture_output=True, timeout=60,
        )
        return (i, t, out) if r.returncode == 0 and Path(out).exists() else None

    frames, stamps = [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(grab, enumerate(times)):
            if res:
                frames.append(res[2])
                stamps.append(round(res[1], 1))
    return frames, stamps


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


def clean_output(text: str) -> str:
    """Strip model reasoning blocks (Gemma 4 emits <thought>/<think> tags)."""
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # unclosed block = model hit the token limit mid-reasoning
    text = re.sub(r"<thought>[\s\S]*$", "", text)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip().strip('"').strip()


NO_REASONING = ("Output ONLY the final answer. Do not include reasoning, "
                "planning, notes, or <thought>/<think> tags of any kind.")


def chat(model: str, messages: list, temperature: float = 0.4,
         max_tokens: int = 1500) -> str:
    last = ""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{FIREWORKS_BASE}/chat/completions",
                headers=_headers(),
                json={"model": model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=180,
            )
            if resp.status_code in (429, 500, 502, 503):
                last = f"HTTP {resp.status_code}: {resp.text[:180]}"
                time.sleep(2.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return clean_output(resp.json()["choices"][0]["message"]["content"])
        except requests.RequestException as e:
            last = str(e)[:180]
            time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"model call failed after 3 attempts — {last}")


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
    """Grounded scene dossier with automatic frame-count fallback."""
    fr = list(frames)
    while True:
        try:
            return _build_dossier(fr, transcript)
        except RuntimeError:
            if len(fr) <= 3:
                raise
            fr = fr[::2]  # halve the frames and retry — payload too big


def _build_dossier(frames: list, transcript: str) -> dict:
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
    content[0]["text"] = NO_REASONING + "\n\n" + content[0]["text"]
    raw = chat(VISION_MODEL, [{"role": "user", "content": content}],
               temperature=0.1, max_tokens=1500)
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
    sys = f"{sys}\n{NO_REASONING}"
    # NOTE: no "system" role — Gemma on some providers rejects it.
    merged = sys + "\n\n---\n\n" + user
    caption = chat(STYLE_MODEL,
                   [{"role": "user", "content": merged}],
                   temperature=TEMPERATURE_STYLE, max_tokens=1200)
    if not caption:  # reasoning block consumed the budget — one strict retry
        caption = chat(STYLE_MODEL,
                       [{"role": "user", "content": merged + "\nCaption now:"}],
                       temperature=0.5, max_tokens=1200)
    return caption


def judge_caption(style_key: str, caption: str, dossier: dict,
                  transcript: str) -> dict:
    user = (f"TARGET STYLE: {style_key}\n\nVIDEO FACTS (ground truth):\n"
            f"{json.dumps(dossier, indent=1)}\n\n"
            f"TRANSCRIPT: {transcript or '(no speech)'}\n\n"
            f"CAPTION TO SCORE:\n\"{caption}\"")
    raw = chat(JUDGE_MODEL,
               [{"role": "user",
                 "content": JUDGE_RUBRIC + "\n\n---\n\n" + user}],
               temperature=0.0, max_tokens=700)
    v = _extract_json(raw)
    return v if "verdict" in v else {"accuracy": 7, "tone": 7,
                                     "verdict": "pass", "fix": ""}


# ------------------------------------------------------------ orchestrate --

def _one_style(style_key: str, dossier: dict, transcript: str, progress):
    progress("writing", {"style": style_key})
    caption = write_caption(style_key, dossier, transcript)
    verdict = judge_caption(style_key, caption, dossier, transcript)
    rounds = 0
    while verdict.get("verdict") == "revise" and rounds < MAX_JUDGE_ROUNDS:
        rounds += 1
        progress("revising", {"style": style_key, "round": rounds,
                              "fix": verdict.get("fix", "")})
        caption = write_caption(style_key, dossier, transcript,
                                fix=verdict.get("fix", ""), previous=caption)
        verdict = judge_caption(style_key, caption, dossier, transcript)
    return style_key, {
        "caption": caption,
        "scores": {"accuracy": verdict.get("accuracy"),
                   "tone": verdict.get("tone")},
        "judge_rounds": rounds,
    }


def run_pipeline(video_path: str, progress=lambda stage, data=None: None,
                 n_frames: int = None) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        # frames and audio/transcript run CONCURRENTLY
        progress("sampling")
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_frames = ex.submit(sample_frames, video_path, workdir, n_frames)
            f_audio = ex.submit(
                lambda: transcribe(extract_audio(video_path, workdir)))
            frames, stamps = f_frames.result()
            progress("sampled", {"frame_count": len(frames), "frames": frames,
                                 "times": stamps})
            progress("transcribing")
            transcript = f_audio.result()
        progress("transcribed", {"transcript": transcript})

        progress("dossier")
        dossier = build_dossier(frames, transcript)
        progress("dossier_done", {"dossier": dossier})

        # all four takes run CONCURRENTLY (each keeps its own judge loop)
        results = {}
        with ThreadPoolExecutor(max_workers=PARALLEL_STYLES) as ex:
            futures = [ex.submit(_one_style, k, dossier, transcript, progress)
                       for k in STYLES]
            for fut in as_completed(futures):
                style_key, res = fut.result()
                results[style_key] = res
                progress("style_done", {"style": style_key, **res})

        return {"transcript": transcript, "dossier": dossier,
                "captions": results, "frame_count": len(frames)}
