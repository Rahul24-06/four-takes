# 🎬 FOUR TAKES — The Screening Room

**AMD Developer Hackathon: ACT II · Track 2 (Video Captioning) · Best Use of Gemma**

One clip. Four voices. A grounded, self-judging caption studio where **Gemma 3 27B**
(via **Fireworks AI**, running on AMD hardware) watches your video and delivers the
same facts in four unmistakable styles: **formal, sarcastic, humorous-tech, and
humorous-non-tech** — each caption scored by an internal LLM judge *before* you
ever see it.

## Why it scores well

Track 2 is judged by an LLM on **accuracy** and **tone**. Four Takes attacks both:

1. **Grounding beats hallucination.** ffmpeg pulls scene-change keyframes and the
   audio track; Whisper transcribes speech; Gemma 3 (multimodal) compiles a
   **scene dossier** — a strict facts-only JSON (subjects, actions in order,
   on-screen text). All four captions are written *from the dossier*, so every
   style agrees on the facts. Accuracy is protected by construction.
2. **Tone is calibrated, not hoped for.** Each style has a named persona,
   a voice spec, a few-shot anchor, and an explicit "avoid" list
   (e.g. non-tech humor bans *all* technical vocabulary — a common failure mode).
3. **Self-judge refinement loop.** Before output, an internal judge scores each
   caption 0–10 on accuracy and tone using a rubric that mirrors the official
   criteria. Captions below 8/8 get sent back with a concrete fix instruction —
   up to 2 retakes. The UI shows the retakes live ("⟲ passed after 1 retake").

## Architecture

```
video ─► ffmpeg keyframes ──┐
      └► ffmpeg audio ─► Whisper (Fireworks)
                            │
                            ▼
              Gemma 3 27B — SCENE DOSSIER (facts-only JSON)
                            │
        ┌──────────┬────────┼──────────┬─────────────┐
        ▼          ▼        ▼          ▼             │
     Formal    Sarcastic  Hum-Tech  Hum-NonTech      │  Gemma 3 27B
        └──────────┴────────┴──────────┘             │  (styling)
                            │                        │
                            ▼                        │
              Gemma 3 27B — LLM JUDGE  ◄─ revise ────┘
              (accuracy ≥8 AND tone ≥8, max 2 retakes)
                            │
                            ▼
                     Four approved takes
```

Every model call goes through **Fireworks AI** (`FIREWORKS_BASE_URL`), and the
whole stack is **Gemma-first** — vision, styling, and judging all on
`gemma-3-27b-it`. Models are env-configurable if the harness restricts choices.

## Quick start (Docker — required by the hackathon)

```bash
cp .env.example .env        # add your FIREWORKS_API_KEY
docker compose up --build
# open http://localhost:8000 — drop a clip, hit ROLL FILM
```

## Batch mode (official clip set)

```bash
mkdir clips && cp /path/to/official/*.mp4 clips/
docker compose run four-takes python scripts/batch.py --clips clips --out results.json
```

`results.json` contains one entry per clip with all four styles plus internal
judge scores. Adjust the output shape in `scripts/batch.py` on launch day when
the official submission format is revealed — the pipeline won't change.

## Configuration

| env var | default | notes |
|---|---|---|
| `FIREWORKS_API_KEY` | — | required |
| `VISION_MODEL` / `STYLE_MODEL` / `JUDGE_MODEL` | `gemma-3-27b-it` | any Fireworks chat model id |
| `WHISPER_MODEL` | `whisper-v3` | Fireworks audio transcription |
| `MAX_FRAMES` | 8 | keyframes per clip |
| `MAX_JUDGE_ROUNDS` | 2 | retakes per style |

## Stack

FastAPI · ffmpeg · Fireworks AI · Gemma 3 27B · Whisper v3 · vanilla JS
(zero frontend build step — one HTML file).

## License

MIT
