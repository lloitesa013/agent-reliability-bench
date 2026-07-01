"""
Hand-crafted traces for the two failure types the offline mock can't produce on
its own (a reasoning error needs a model that reasons; an ambiguous case needs a
genuinely borderline question). These exercise the watcher's harder paths so we
can verify it (a) catches a wrong answer even when retrieval was fine, and
(b) ABSTAINS when the evidence is unclear.

With a real LLM these arise naturally; here we inject them as labeled fixtures.
"""

import json
from pathlib import Path

CRAFTED = [
    # reasoning_error: the right context WAS retrieved, but the agent answered wrong (1889 -> 1989)
    {
        "question": "When was the Eiffel Tower completed?",
        "steps": [
            {"step": "retrieve", "retrieved": [
                "The Eiffel Tower is located in Paris, France, and was completed in 1889."]},
            {"step": "answer", "answer": "The Eiffel Tower was completed in 1989."},
        ],
        "answer": "The Eiffel Tower was completed in 1989.",
        "expected": "1889",
        "ground_truth_cause": "reasoning_error",
    },
    # uncertain: vague target, only partially supported -> the watcher should ABSTAIN
    {
        "question": "What is Everest known as?",
        "steps": [
            {"step": "retrieve", "retrieved": [
                "Mount Everest is the highest mountain above sea level, at 8,849 meters."]},
            {"step": "answer", "answer": "Mount Everest is the highest mountain above sea level."},
        ],
        "answer": "Mount Everest is the highest mountain above sea level.",
        "expected": "Everest tallest peak",
        "ground_truth_cause": "uncertain",
    },
]

if __name__ == "__main__":
    out = Path("traces")
    out.mkdir(exist_ok=True)
    for i, t in enumerate(CRAFTED, start=8):
        (out / f"trace_{i}.json").write_text(json.dumps(t, indent=2))
    print(f"Wrote {len(CRAFTED)} crafted hard-case traces to ./traces/")
