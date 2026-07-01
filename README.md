# Agent Reliability Demo

A small, working tool that **diagnoses *why* an LLM/RAG agent failed** and reports
**how much to trust that diagnosis** — making agent failures *legible and diagnosable*
(trustworthy AI). Practical, hiring-relevant (LLM agent / RAG reliability).

## What it does
1. `agent.py` — a tiny RAG agent (retrieve → answer) that logs a **trace** of every step.
2. `diagnose.py` — given a trace, attributes the failure cause
   (`retrieval_miss` / `missing_knowledge` / `reasoning_error` / `correct_abstention`)
   with a **calibrated confidence**, and **abstains (`uncertain`)** when the evidence is unclear.
3. `evaluate.py` — measures the diagnoser against ground truth: accuracy, abstention rate,
   and **calibration** (when it's confident, is it actually right?).

> The point: it doesn't just guess a cause — it says how much to trust the guess, and
> stays silent when it can't tell. That "knows when it's unsure" property is the core idea.

## Run it (works offline — no API key needed)
```bash
python agent.py        # produces ./traces/*.json   (mock LLM by default)
python diagnose.py     # prints a diagnosis + confidence for each trace
python evaluate.py     # accuracy / abstention / calibration summary
```
For a **real LLM run**: Python 3.8+, `pip install openai`, set `OPENAI_API_KEY`
(or swap to Anthropic in `agent.py`). Offline mock runs on older Python too.

## Roadmap
- [x] Step 1 — trace
- [x] Step 2 — diagnose (calibrated confidence + abstain)
- [x] Step 3 — evaluate (accuracy / abstention / calibration)
- [ ] Step 4 — real LLM run, more failure types, real retrieval (vector DB), cleaner report

> Honest scope: competent execution on a real problem (agent reliability) — not a novel
> SOTA claim. Value = a working, legible, *self-aware* reliability tool + honest evaluation.

## License
MIT
