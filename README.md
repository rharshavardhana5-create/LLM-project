# CodeReviewBot — Fine-Tuned LLM for Automated Code Review Comments

A QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct that reads a code diff and
writes the kind of review comment a senior engineer would leave —
including knowing when to say nothing at all.

## Why this project

Most "fine-tune an LLM" portfolio projects stop at "loss went down." This
one is built to hold up in an interview: it has a real evaluation (not
vibes), it handles the boring-but-important case of clean code that needs
no comment, and it ships as something runnable, not just a training log.

## Project structure

| File | Purpose |
|---|---|
| `train_qlora.ipynb` | Colab notebook — self-contained, run top to bottom on a free T4 |
| `data_prep.py` | Standalone version of the data step, for reproducibility outside Colab |
| `evaluate.py` | Baseline vs. fine-tuned comparison: ROUGE-L, negative-detection accuracy, LLM-as-judge |
| `serve.py` | Merge the adapter and serve it (CLI or local FastAPI endpoint) |
| `requirements.txt` | Dependencies for `evaluate.py` / `serve.py` |

## Dataset

[`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) —
167K+ real `(before_code, reviewer_comment, after_code)` triplets from 725
top-starred GitHub repos, plus ~51K "clean code, no comment" negatives.
We filter to Python and a quality threshold, and keep ~15% negatives in
the final mix so the model learns restraint, not just fault-finding.

## Base model

`unsloth/Qwen2.5-Coder-7B-Instruct` (4-bit), fine-tuned with QLoRA via
Unsloth + TRL's `SFTTrainer`. Loss is masked to the assistant's response
tokens only (`train_on_responses_only`) — worth calling out in an
interview, since naive SFT that trains on the full prompt+response wastes
capacity and biases generation toward echoing the input.

## How to run

1. **Train** — open `train_qlora.ipynb` in Google Colab
   (`Runtime → Change runtime type → T4 GPU`), then Runtime → Run all.
   Roughly 45–90 minutes on a free-tier T4 depending on how many examples
   you keep (`MAX_TRAIN` in the notebook).
2. **Evaluate** — download the saved `qwen_codereview_lora` adapter
   folder, then on a machine with a GPU:
   ```bash
   pip install -r requirements.txt
   ANTHROPIC_API_KEY=sk-... python evaluate.py --adapter qwen_codereview_lora
   ```
3. **Serve** —
   ```bash
   python serve.py --code some_file.py     # one-off CLI review
   python serve.py --serve                  # local API on :8000
   ```

## Results

*(Fill in after your run — this table is what you'll actually talk
through in an interview.)*

| Metric | Baseline (zero-shot) | Fine-tuned |
|---|---|---|
| ROUGE-L vs. human comment | | |
| Negative-detection accuracy | | |
| False positive rate (flags clean code) | | |
| False negative rate (misses real issues) | | |
| LLM-judge: correctness (1–5) | | |
| LLM-judge: specificity (1–5) | | |
| LLM-judge: tone (1–5) | | |

## Talking points for interviews

- **Why QLoRA over full fine-tuning** — 7B params don't fit for full
  fine-tuning on a single T4; 4-bit quantization + low-rank adapters get
  ~0.1–1% of parameters trainable with comparable quality on a narrow task.
- **Why loss masking matters** — without `train_on_responses_only`, the
  model spends gradient steps learning to predict the prompt it was just
  given, which is wasted capacity and can bias generation.
- **Why negatives are in the training set** — a review bot that always
  finds something to say is worse than useless in a real PR pipeline. The
  false-positive rate is the metric that would actually block shipping this.
- **Why an LLM judge, not just ROUGE** — ROUGE rewards surface overlap
  with one reference comment, but two very different phrasings can both
  be excellent reviews. The judge rubric scores what you actually care
  about — correctness, specificity, tone — independent of wording.
- **What you'd do next** — scale to more languages, add repo-level
  context via retrieval instead of a fixed 50-line chunk, or wrap this as
  an agent that posts the comment via the GitHub API and can respond to
  follow-up pushback.

## Known limitations

- Single ~50-line chunk of context — no whole-file or cross-file reasoning.
- Trained on Python only; multi-language needs a bigger training set per
  language or a smarter mixing strategy.
- The LLM judge introduces its own bias — worth naming as a limitation
  rather than hiding it if it comes up.
