# PathForge — Evaluation

Run: `python -m src.evaluation`

Reports (on real, hand-labeled sample data — see src/evaluation.py):
1. Skill extraction precision / recall / F1 against a manually labeled resume snippet
2. Semantic retrieval Top-K relevance for an obviously-matching query
3. Skill coverage formula sanity check against a hand-computed example
4. Recommendation ranking consistency (same input -> same output)

These are intentionally small and transparent rather than large benchmark
claims, matching the project's "explainable, viva-ready" requirement.
