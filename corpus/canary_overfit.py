#!/usr/bin/env python3
"""
canary_overfit.py — 20-second "can this stack learn AT ALL?" test.

WHY THIS EXISTS
---------------
On 1 June 2026 the full retrain collapsed: loss flat at ~0.65, eval_f1_micro=0.0
from the first epoch, model predicted "absent" for everything. That signature means
the model never learned — but a 30-minute run to discover that is wasteful, and it
does not tell you WHERE the break is (precision? library stack? optimisation?).

This canary tries to OVERFIT a tiny batch (default 16 rows) with a transparent manual
training loop in fp32. A healthy transformer + autograd stack overfits 16 examples to
near-zero loss in ~100 steps, trivially. The outcome discriminates:

  * Loss drives toward 0 AND predictions stop being all-identical
        -> the MACHINERY is fine. If the full run still collapses, the cause is
           optimisation dynamics (LR/scale/precision in the Trainer path), not the stack.

  * Loss stays flat (~0.69) and predictions are frozen / all-identical
        -> the MACHINERY is broken (precision kernels, library/CUDA mismatch, bad
           tokenisation). A full run cannot help. Next step: pin the January stack
           (see RELABEL-RUNBOOK / RUN-V2). Do NOT launch the 30-minute run.

Deliberately uses a plain torch loop (no Trainer/accelerate) so it isolates the core
model forward/backward under the precision you pass. Default precision: fp32.

Exit code: 0 = learned (PASS), 3 = did not learn (FAIL / machinery broken).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DIMS = ["CLAIM", "EVIDENCE", "SCOPE", "ASSUMPTIONS", "GAPS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--n", type=int, default=16, help="rows to overfit")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4, help="generous LR — we WANT overfit")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pass_loss", type=float, default=0.20,
                    help="final loss below this = learned")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    print("=" * 70)
    print(f"[canary] python {platform.python_version()} | torch {torch.__version__} "
          f"| transformers {transformers.__version__} | cuda {torch.version.cuda}")
    print(f"[canary] device={device} | precision={precision} | n={args.n} "
          f"| steps={args.steps} | lr={args.lr}")
    print("=" * 70, flush=True)

    rows = [json.loads(l) for l in open(args.train_data) if l.strip()][: args.n]
    texts = [r["text"] for r in rows]
    labels = torch.tensor([r["labels"] for r in rows], dtype=torch.float, device=device)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    enc = tok(texts, return_tensors="pt", truncation=True,
              max_length=args.max_length, padding="max_length").to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=5, problem_type="multi_label_classification"
    ).to(device)
    model.train()

    # Mixed precision via autocast only when requested; fp32 is a plain loop.
    use_amp = precision in ("bf16", "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossfn = torch.nn.BCEWithLogitsLoss()

    losses = []
    for step in range(args.steps):
        opt.zero_grad()
        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"]).logits
                loss = lossfn(logits.float(), labels)
            if precision == "fp16":
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).logits
            loss = lossfn(logits, labels)
            loss.backward(); opt.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {float(loss):.4f}", flush=True)

    # --- Evaluate the overfit: loss, prediction variety, train accuracy ----------
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

    final_loss = losses[-1]
    pred_pos_frac = float(preds.float().mean())
    # "frozen / all-identical": every row predicted the same 5-bit pattern
    unique_rows = len(set(tuple(r.tolist()) for r in preds))
    train_acc = float((preds == labels.int()).float().mean())
    nan_logits = bool(torch.isnan(logits).any())

    print("\n" + "-" * 70)
    print(f"[canary] final_loss={final_loss:.4f}  train_acc={train_acc:.3f}  "
          f"pred_pos_frac={pred_pos_frac:.3f}  unique_pred_rows={unique_rows}/{args.n}  "
          f"nan_logits={nan_logits}")

    learned = (final_loss < args.pass_loss) and (unique_rows > 1) and (not nan_logits)
    if learned:
        print(f"[canary] PASS — the stack CAN learn (overfit {args.n} rows in fp32-loop). "
              f"If the full run still collapses, look at LR/precision/Trainer dynamics, "
              f"not the stack.")
        print("-" * 70, flush=True)
        sys.exit(0)
    else:
        why = []
        if final_loss >= args.pass_loss: why.append(f"loss never dropped (>{args.pass_loss})")
        if unique_rows <= 1: why.append("predictions frozen/all-identical")
        if nan_logits: why.append("NaN logits")
        print(f"[canary] FAIL — MACHINERY BROKEN: {', '.join(why)}.")
        print(f"[canary] A full 30-min run will NOT help. Likely the latest-everything "
              f"venv/CUDA vs DeBERTa-v3. Next: pin the January stack (see RUN-V2 notes), "
              f"or try a different precision (--bf16 / --fp16) to localise it.")
        print("-" * 70, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
