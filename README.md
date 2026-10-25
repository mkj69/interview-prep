# Interview Prep

A recall-first notebook for **Research Scientist internship interviews** (LLMs, post-training/RL, reasoning, agents, systems, ML coding, research discussion).

Live site: https://mkj69.github.io/interview-prep/

## How it is built

- Plain static HTML/CSS/JS, served by GitHub Pages. No build step.
- `assets/modules.js` — the roadmap (module list, priorities, status).
- `modules/mNN-*.html` — one page per module. Each topic follows the same layers: one-line answer (中文 + English) → explanation → derivation/code (folded) → "Say it in English" → follow-up chain.
- `assets/cards/mNN.js` — flashcards for each module.
- `assets/app.js` — speed mode (hides everything except one-liners and English answers), TOC, copy buttons, KaTeX/highlight.js, and a small Leitner-style spaced-repetition scheduler. Review progress is stored only in the reader's browser (`localStorage`).
- `review.html` — daily review queue across all modules (keyboard: space to reveal, 1/2/3 to rate).

Explanations are in Chinese; English answers are written to be spoken aloud in interviews.

## Verifying the PyTorch code

The site was written in a sandbox without PyTorch, so every PyTorch block was checked against an equivalent NumPy implementation. The PyTorch blocks themselves are saved as standalone tests in `tests/torch/` (82 files). On any machine with PyTorch:

```bash
pip install torch numpy
python tests/run_torch_tests.py
```
