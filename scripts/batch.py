#!/usr/bin/env python3
"""
Batch runner for the official fixed clip set.

    python scripts/batch.py --clips ./clips --out results.json

Produces one JSON object per clip with all four styles. Adjust the output
shape here on launch day once the official submission format is revealed —
the pipeline itself won't need to change.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import run_pipeline  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="directory of video clips")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    clips = sorted(p for p in Path(args.clips).iterdir()
                   if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        sys.exit(f"No video files found in {args.clips}")

    results = {}
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip.name}", flush=True)
        out = run_pipeline(str(clip),
                           progress=lambda s, d=None: print(f"   · {s}", flush=True))
        results[clip.name] = {
            "formal": out["captions"]["formal"]["caption"],
            "sarcastic": out["captions"]["sarcastic"]["caption"],
            "humorous_tech": out["captions"]["humorous_tech"]["caption"],
            "humorous_non_tech": out["captions"]["humorous_non_tech"]["caption"],
            "_meta": {"internal_scores": {k: v["scores"]
                                          for k, v in out["captions"].items()}},
        }
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
