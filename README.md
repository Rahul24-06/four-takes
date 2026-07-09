# 🎬 FOUR TAKES — The Screening Room

**AMD Developer Hackathon: ACT II · Track 2 (Video Captioning) · Best Use of Gemma**

One clip. Four voices. A grounded, self-judging caption studio where **Gemma 3 27B**
(via **Fireworks AI**, running on AMD hardware) watches your video and delivers the
same facts in four unmistakable styles: **formal, sarcastic, humorous-tech, and
humorous-non-tech** — each caption scored by an internal LLM judge *before* you
ever see it.


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

## Quick start

```bash
cp .env.example .env        
docker compose up --build
# open http://localhost:8000 — drop a clip, hit ROLL FILM
```

## Stack

FastAPI · ffmpeg · Fireworks AI · Gemma 3 27B · Whisper v3 · vanilla JS


## License

MIT
