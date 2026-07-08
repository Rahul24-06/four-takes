"""
Style definitions, few-shot anchors, and judge rubrics.

The official Track 2 scoring is an LLM-judge on ACCURACY and TONE.
Two levers matter most:
  1. Grounding  -> captions must reference what actually happens (frames + audio).
  2. Tone match -> each style must be unmistakable, not generically "funny".

Few-shot anchors below are style exemplars ONLY (about a fictional placeholder
clip) so the model copies the VOICE, never the content.
"""

STYLES = {
    "formal": {
        "label": "The Anchor",
        "voice": (
            "Neutral broadcast-register English. Third person, present tense. "
            "Precise nouns and verbs, zero slang, zero jokes, zero opinions. "
            "Reads like a news wire summary or an archival catalogue entry."
        ),
        "anchor": (
            "A technician assembles a small wheeled robot on a workbench, "
            "connecting a motor driver board before testing forward motion. "
            "The device completes a short straight-line run as an observer "
            "records the result."
        ),
        "avoid": "humor, exclamation marks, first person, rhetorical questions",
    },
    "sarcastic": {
        "label": "The Cynic",
        "voice": (
            "Dry, deadpan, faintly unimpressed. The humor comes from ironic "
            "understatement and mock-praise, NOT from insults or meanness. "
            "Eye-roll energy; every compliment has a raised eyebrow in it."
        ),
        "anchor": (
            "Ah yes, another robot conquering the legendary challenge of "
            "driving in a straight line. Truly, the machines are coming for "
            "our jobs — assuming our jobs involve rolling three feet and "
            "stopping. Riveting stuff."
        ),
        "avoid": "cruelty, profanity, being sincere, tech jargon overload",
    },
    "humorous_tech": {
        "label": "The Debugger",
        "voice": (
            "Jokes built FROM technical culture: bugs, git, deadlines, Stack "
            "Overflow, off-by-one errors, 'works on my machine'. The viewer "
            "should need some tech literacy to fully get the joke. Warm, "
            "insider humor — laughing WITH engineers."
        ),
        "anchor": (
            "Robot drives straight on the first try — clearly a staged demo, "
            "because no hardware works before the third firmware flash. "
            "Somewhere a PID loop is taking full credit for what was honestly "
            "just a flat table. Ship it before it learns about edge cases."
        ),
        "avoid": "jargon with no joke attached, sarcasm-only tone, meanness",
    },
    "humorous_non_tech": {
        "label": "The Standup",
        "voice": (
            "Broad, accessible, observational comedy anyone's grandmother "
            "would get. Everyday analogies, playful exaggeration, physical "
            "comedy framing. Zero technical vocabulary — if a word needs a "
            "manual, it's out."
        ),
        "anchor": (
            "This little robot rolls across the table with the confidence of "
            "a toddler who just learned to walk and immediately headed for "
            "the good china. Everyone watches it like proud parents at a "
            "school play — three feet of travel, standing ovation."
        ),
        "avoid": "any technical terminology, sarcasm, insider references",
    },
}

# Rubric the internal judge uses — deliberately mirrors "accuracy and tone".
JUDGE_RUBRIC = """You are a strict evaluation judge for video captions.
Score the caption on two axes, each 0-10:

ACCURACY: Does the caption faithfully reflect the actual video content
(subjects, actions, setting, spoken audio, on-screen text)? Penalize any
invented detail (hallucination) heavily. Penalize vagueness that could
describe any video.

TONE: Does the caption land squarely in the target style?
- formal: neutral, precise, journalistic; ANY humor = fail.
- sarcastic: dry irony and mock-praise; sincere or mean = fail.
- humorous_tech: joke requires tech-culture literacy; jargon without a
  joke = fail; joke a non-techie fully gets = weak.
- humorous_non_tech: universally accessible comedy; ANY technical
  vocabulary = fail.

Also check: caption length 1-4 sentences, self-contained, no meta-text
("this video shows..." is acceptable for formal only, weak elsewhere).

Reply ONLY with JSON:
{"accuracy": <0-10>, "tone": <0-10>, "verdict": "pass"|"revise",
 "fix": "<one concrete instruction if revise, else empty string>"}
A caption passes only if accuracy >= 8 AND tone >= 8."""
