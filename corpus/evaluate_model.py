#!/usr/bin/env python3
"""
evaluate_model.py — honest before/after evaluation for the relabel+retrain.

WHY THIS EXISTS
---------------
train_stage1.py reports accuracy against the labels it trained on. Retrain on the
corrected labels and it will ALSO report ~98% — because the model fits whatever
labels you give it. That number is NOT a before/after and must not be read as one.

The honest test is: score the OLD model and the NEW model against the SAME
corrected test set (val.relabel.deberta.jsonl), and on a few fixed probe inputs.
Run this script twice — once per model — and compare.

  python evaluate_model.py --model ../models/deberta-coherence            --data val.relabel.deberta.jsonl   # OLD
  python evaluate_model.py --model model-v1.1-relabel                     --data val.relabel.deberta.jsonl   # NEW

The number that matters is EVIDENCE / ASSUMPTIONS accuracy against the CORRECTED
labels, and the "predicted PRESENT when label says ABSENT" over-prediction rate.
That over-prediction rate is the bug Pablo found; it should fall sharply for the
new model.

CPU is fine (val is 723 rows); a GPU box is faster.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DIMS = ["CLAIM", "EVIDENCE", "SCOPE", "ASSUMPTIONS", "GAPS"]

# Fixed probes — the vivid before/after. Pablo's failure was a fluent, evidence-free,
# implicit-assumption concept scoring EVIDENCE/ASSUMPTIONS as present.
PROBES = [
    ("fluent, NO evidence, IMPLICIT assumption (Pablo-shape)",
     "Our platform helps students verify their own reasoning before they submit work, "
     "making critical thinking a daily habit across the university."),
    ("same claim WITH real evidence",
     "In a pilot with 120 students at UNIR, a self-verification step reduced unsupported "
     "claims by 34 percent (Martinez, 2026); we build on that finding."),
    ("own-programme SCALE only (prior work, not evidence)",
     "Our programme already works with 1,200 families across 25 villages in the district."),
    ("EXPLICIT assumption stated",
     "This works only where students have laptop access; we assume a stable campus network."),
]


def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tok, model


@torch.no_grad()
def scores(tok, model, texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
        logits = model(**enc).logits
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir")
    ap.add_argument("--data", required=True, help="deberta-form jsonl: {text, labels:[5]}")
    ap.add_argument("--json", default="", help="optional path to write machine-readable metrics")
    args = ap.parse_args()

    tok, model = load_model(args.model)

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    texts = [r["text"] for r in rows]
    y = np.array([r["labels"] for r in rows])           # (N,5) gold (corrected) labels
    p = (scores(tok, model, texts) > 0.5).astype(int)   # (N,5) predictions

    print(f"\nMODEL: {args.model}")
    print(f"TEST SET: {args.data}  (n={len(rows)}, corrected labels)\n")
    print(f"{'dim':12} {'acc':>7} {'pred-PRESENT-when-ABSENT':>26}")
    per_dim = {}
    for i, d in enumerate(DIMS):
        acc = (p[:, i] == y[:, i]).mean()
        # over-prediction: label says 0 (absent) but model says 1 (present) — the bug
        absent = (y[:, i] == 0)
        over = (p[absent, i] == 1).mean() if absent.any() else float("nan")
        flag = "  <-- the bug" if d in ("EVIDENCE", "ASSUMPTIONS") else ""
        print(f"{d:12} {acc:7.3f} {over:26.3f}{flag}")
        per_dim[d] = {"acc": float(acc), "over_predict_when_absent": float(over),
                      "pred_pos": int(p[:, i].sum()), "gold_pos": int(y[:, i].sum())}
    mean_acc = float((p == y).mean())
    f1 = float(f1_score(y, p, average="micro"))
    pred_pos_frac = float(p.mean())
    print(f"\nmean acc: {mean_acc:.3f}")

    # Collapse check, printed explicitly so a dead model is unmistakable.
    print("\npredicted-positive count per dim (collapse check):")
    for d in DIMS:
        print(f"  {d:12} pred+={per_dim[d]['pred_pos']:4d}/{len(rows)}   "
              f"gold+={per_dim[d]['gold_pos']:4d}")
    print(f"\nf1_micro: {f1:.3f}   total pred-positive fraction: {pred_pos_frac:.3f}"
          + ("   *** COLLAPSED (predicts ~nothing present) ***" if pred_pos_frac < 1e-6 or f1 == 0.0 else ""))

    print("\n--- PROBES (raw sigmoid; EVIDENCE=index1, ASSUMPTIONS=index3) ---")
    ps = scores(tok, model, [t for _, t in PROBES])
    probe_out = []
    for (label, _), s in zip(PROBES, ps):
        print(f"  EV={s[1]:.2f}  AS={s[3]:.2f}   {label}")
        probe_out.append({"label": label, "EV": float(s[1]), "AS": float(s[3])})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "model": args.model, "data": args.data, "n": len(rows),
                "mean_acc": mean_acc, "f1_micro": f1, "pred_pos_frac": pred_pos_frac,
                "collapsed": bool(pred_pos_frac < 1e-6 or f1 == 0.0),
                "per_dim": per_dim, "probes": probe_out,
            }, fh, indent=2)
        print(f"\n[json] metrics -> {args.json}")


if __name__ == "__main__":
    main()
