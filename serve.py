"""
serve.py

Two ways to use the fine-tuned code reviewer:

1. One-off CLI review of a file:
    python serve.py --code path/to/snippet.py

2. Local API server:
    python serve.py --serve
    curl -X POST localhost:8000/review -H "Content-Type: application/json" \
         -d '{"code": "def add(a,b):\n  return a+b", "language": "Python"}'

Loads the base model + LoRA adapter and merges them once at startup
(merge_and_unload) so inference doesn't pay adapter overhead per request.
"""

import argparse

import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert code reviewer. You will be shown a chunk of code "
    "from a pull request, with its language and file path. Write the "
    "single most important review comment a senior engineer would leave. "
    "Be specific and actionable. If the code has no issues worth "
    "flagging, respond with exactly: No issues found."
)


def load_model(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def review(model, tokenizer, code: str, language: str = "Python", file_path: str = "snippet.py") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"File: {file_path}\nLanguage: {language}\n\nCode chunk:\n{code}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ReviewRequest(BaseModel):
    code: str
    language: str = "Python"
    file_path: str = "snippet.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="qwen_codereview_lora")
    parser.add_argument("--code", help="Path to a code file to review (CLI mode)")
    parser.add_argument("--serve", action="store_true", help="Start a local FastAPI server instead")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.serve:
        app = FastAPI(title="Code Review Assistant")

        @app.post("/review")
        def review_endpoint(req: ReviewRequest):
            comment = review(model, tokenizer, req.code, req.language, req.file_path)
            return {"comment": comment}

        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.code:
        with open(args.code) as f:
            code = f.read()
        print(review(model, tokenizer, code, file_path=args.code))
    else:
        parser.error("Pass --code <file> for a one-off review, or --serve to start the API.")


if __name__ == "__main__":
    main()
