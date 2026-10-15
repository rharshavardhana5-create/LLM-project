"""
evaluate.py

Compares the zero-shot base model against your fine-tuned LoRA adapter on
the held-out test set, using three complementary signals:

  1. ROUGE-L        — surface overlap with the human reviewer's comment.
                       Weak on its own (many good phrasings exist) but cheap
                       and useful as a sanity check.
  2. Negative-detection accuracy — did the model correctly say
                       "No issues found." on clean code, and correctly NOT
                       say that on code that actually needed a comment?
                       Reported as accuracy + false-positive/false-negative
                       rate, because these two error types matter very
                       differently in a real review pipeline.
  3. LLM-as-judge    — Claude scores each generated comment 1-5 on
                       correctness, specificity, and tone, using the human
                       comment as a reference (not a rubric to string-match
                       against).

Run this on a GPU machine after training and downloading the adapter
folder produced by train_qlora.ipynb:

    pip install -r requirements.txt
    ANTHROPIC_API_KEY=sk-... python evaluate.py --adapter qwen_codereview_lora
"""

import argparse
import json
import os
import random
import statistics as stats

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
NO_ISSUES = "no issues found"

JUDGE_RUBRIC = """You are grading an AI-generated code review comment against a
reference comment written by a real human reviewer, for the same code chunk.

Code chunk:
{code}

Human reviewer's comment (reference):
{reference}

AI-generated comment:
{prediction}

Score the AI comment from 1-5 on each axis:
- correctness: does it identify a real, relevant issue (or correctly say "No issues found")?
- specificity: is it concrete and actionable, not generic boilerplate?
- tone: does it read like a real senior engineer, not a robotic disclaimer?

Respond with ONLY a JSON object, no other text:
{{"correctness": <int>, "specificity": <int>, "tone": <int>, "rationale": "<one sentence>"}}
"""


def load_test_set(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate(model, tokenizer, messages, max_new_tokens=150):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def negative_detection_stats(rows, preds):
    tp = tn = fp = fn = 0
    for row, pred in zip(rows, preds):
        said_no_issues = NO_ISSUES in pred.lower()
        is_negative = row["meta"]["is_negative"]
        if is_negative and said_no_issues:
            tn += 1
        elif is_negative and not said_no_issues:
            fp += 1  # flagged an issue on clean code
        elif not is_negative and said_no_issues:
            fn += 1  # missed a real issue
        else:
            tp += 1
    total = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def rouge_l(rows, preds):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for row, pred in zip(rows, preds):
        ref = row["messages"][2]["content"]
        scores.append(scorer.score(ref, pred)["rougeL"].fmeasure)
    return stats.mean(scores) if scores else 0.0


def llm_judge(rows, preds, model_name="claude-haiku-4-5-20251001", sample_size=60):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    idx = list(range(len(rows)))
    random.shuffle(idx)
    idx = idx[:sample_size]

    results = []
    for i in idx:
        row, pred = rows[i], preds[i]
        code = row["messages"][1]["content"]
        reference = row["messages"][2]["content"]
        prompt = JUDGE_RUBRIC.format(code=code[:2000], reference=reference, prediction=pred)
        resp = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        try:
            results.append(json.loads(text))
        except json.JSONDecodeError:
            continue

    if not results:
        return {}
    return {axis: stats.mean(r[axis] for r in results) for axis in ["correctness", "specificity", "tone"]}


def run_model(label, model, tokenizer, rows):
    print(f"\nGenerating with {label}...")
    preds = [generate(model, tokenizer, row["messages"][:2]) for row in rows]

    print(f"  ROUGE-L:              {rouge_l(rows, preds):.3f}")
    neg = negative_detection_stats(rows, preds)
    print(f"  Negative accuracy:    {neg['accuracy']:.3f}")
    print(f"  False positive rate:  {neg['false_positive_rate']:.3f}  (flagged clean code)")
    print(f"  False negative rate:  {neg['false_negative_rate']:.3f}  (missed a real issue)")

    if os.environ.get("ANTHROPIC_API_KEY"):
        judged = llm_judge(rows, preds)
        for axis, score in judged.items():
            print(f"  LLM-judge {axis:<12}: {score:.2f} / 5")
    else:
        print("  (set ANTHROPIC_API_KEY to also run the LLM-judge scoring)")
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="qwen_codereview_lora")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--base_model", default=BASE_MODEL)
    args = parser.parse_args()

    rows = load_test_set(args.test_file)
    print(f"Loaded {len(rows)} test examples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    print("\n### Baseline (zero-shot) ###")
    run_model("baseline", base_model, tokenizer, rows)

    print("\n### Fine-tuned (LoRA) ###")
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter)
    run_model("fine-tuned", tuned_model, tokenizer, rows)


if __name__ == "__main__":
    main()
