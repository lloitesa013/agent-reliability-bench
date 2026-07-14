"""
A tiny RAG agent that logs its trace. Runs OFFLINE by default (mock LLM, no key
needed); set OPENAI_API_KEY to use a real model.
"""

import json
import os
from pathlib import Path

_warned_fallback = False  # one-time notice when a real-LLM run degrades to the mock

KNOWLEDGE = [
    "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    "Mount Everest is the highest mountain above sea level, at 8,849 meters.",
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Great Wall of China is over 13,000 miles long.",
    "The Eiffel Tower replica in Las Vegas is about half the height of the original.",  # distractor
    "The capital of France is Paris.",  # distractor: hijacks 'capital of ...' questions
    "Tokyo is the seat of government and the largest city of Japan.",  # the real answer, low keyword overlap
]


def retrieve(question, k=1):
    def overlap(doc):
        return len(set(question.lower().split()) & set(doc.lower().split()))
    return sorted(KNOWLEDGE, key=overlap, reverse=True)[:k]


def call_llm(prompt):
    # Use a real model if a key is set AND the SDK is installed; otherwise fall back
    # to the offline mock so the pipeline always runs.
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip()
        except Exception as e:
            # never degrade silently: a mock run must not pass as a real-LLM run
            global _warned_fallback
            if not _warned_fallback:
                _warned_fallback = True
                print("[warn] OpenAI call failed (%s) -> falling back to offline mock for this run" % type(e).__name__)
    return _mock_llm(prompt)


def _mock_llm(prompt):
    """Offline stand-in so the pipeline runs with no API key. Extracts the most
    relevant context line; says IDK when the context looks irrelevant."""
    ctx = prompt.split("Context:")[1].split("Question:")[0]
    q = prompt.split("Question:")[1].split("Answer:")[0]
    qwords = {w.strip("?.,").lower() for w in q.split()}
    best, score = "", 0
    for line in ctx.strip().splitlines():
        s = len(qwords & {w.strip(".,").lower() for w in line.split()})
        if s > score:
            best, score = line, s
    return best.strip() if score >= 2 else "I don't know."


def run_agent(question):
    trace = {"question": question, "steps": []}
    retrieved = retrieve(question)
    trace["steps"].append({"step": "retrieve", "retrieved": retrieved})
    prompt = (
        "Answer the question using ONLY the context. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{chr(10).join(retrieved)}\n\nQuestion: {question}\nAnswer:"
    )
    trace["steps"].append({"step": "prompt", "prompt": prompt})
    answer = call_llm(prompt)
    trace["steps"].append({"step": "answer", "answer": answer})
    trace["answer"] = answer
    return trace


# (question, expected_answer, ground_truth_cause) -- ground truth used by evaluate.py
CASES = [
    ("Where is the Eiffel Tower?", "Paris", "no_failure"),
    ("Who created Python?", "Guido van Rossum", "no_failure"),
    ("What is the capital of Japan?", "Tokyo", "retrieval_miss"),   # Tokyo IS in KB, but the France distractor gets retrieved
    ("Who wrote Hamlet?", "Shakespeare", "missing_knowledge"),      # not in the KB at all
]

if __name__ == "__main__":
    out = Path("traces")
    out.mkdir(exist_ok=True)
    for i, (q, expected, gt) in enumerate(CASES):
        t = run_agent(q)
        t.update(expected=expected, ground_truth_cause=gt)
        t["looks_correct"] = expected.lower() in t["answer"].lower()
        (out / f"trace_{i}.json").write_text(json.dumps(t, indent=2))
        print(f"[{'OK ' if t['looks_correct'] else 'FAIL'}] {q}\n        -> {t['answer']}\n")
    print("Traces saved to ./traces/ . Now run:  python diagnose.py   and   python evaluate.py")
