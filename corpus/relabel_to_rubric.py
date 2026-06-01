#!/usr/bin/env python3
"""
relabel_to_rubric.py — strict-rubric relabelling of EVIDENCE and ASSUMPTIONS.

WHY THIS EXISTS
---------------
A 2026 audit of this corpus found that the EVIDENCE and ASSUMPTIONS labels drifted
looser than the tool's own rubric (spec.md):

  - 64% of texts labelled EVIDENCE=1 reference no observation/research/data/prior work.
  - 71% of texts labelled ASSUMPTIONS=1 name no explicit dependency or condition.

The DeBERTa model faithfully reproduced that drift, so fluent-but-empty concepts
scored as "evidence present" / "assumptions acknowledged". The fix is upstream:
re-derive ONLY the EVIDENCE and ASSUMPTIONS labels by applying the spec.md rubric
strictly, then retrain. CLAIM, SCOPE and GAPS labels are left untouched.

This script does the relabel. It does NOT retrain. After relabelling and a HUMAN
spot-check of the diff, retrain with train_stage1.py (see RELABEL-RUNBOOK.md).

WHAT IT GUARANTEES
------------------
  - Deterministic: temperature=0, fixed model.
  - Idempotent / resumable: per-record cache keyed by sha1(text). Re-runs skip done work.
  - Auditable: emits a diff CSV (every flip, with the model's one-line reason) for spot-check.
  - Drop-in: emits the list-form deberta_*.jsonl the trainer consumes, plus the dict form.

USAGE
-----
  export OPENROUTER_API_KEY=...
  # Validate on a sample first (cheap — ~50 records):
  python relabel_to_rubric.py --input train.jsonl --out-prefix train.relabel --sample 50
  # Inspect train.relabel.diff.csv by hand, then run the full pass:
  python relabel_to_rubric.py --input train.jsonl --out-prefix train.relabel
  python relabel_to_rubric.py --input val.jsonl   --out-prefix val.relabel
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

DIMENSION_ORDER = ["CLAIM", "EVIDENCE", "SCOPE", "ASSUMPTIONS", "GAPS"]
MODEL = "anthropic/claude-haiku-4.5"

# ---------------------------------------------------------------------------
# The rubric, transcribed VERBATIM from spec.md (the EVIDENCE and ASSUMPTIONS
# dimension definitions). This is the contract. Do not paraphrase it here —
# if spec.md changes, change this to match.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict annotator for a design-concept coherence corpus.
You decide TWO binary labels for a short design-concept text: EVIDENCE and ASSUMPTIONS.
You apply the rubric below literally. You are deliberately conservative.

OVERRIDING RULE — judge only what the text ACTUALLY CONTAINS.
Fluent, confident, well-structured, plausible-sounding writing is NOT evidence and is
NOT an acknowledged assumption. Do not infer that evidence exists because a claim sounds
credible. Do not infer that an assumption is acknowledged because you can see what the
assumption would be. If the text only asserts, the label is 0.

================================================================================
DIMENSION: EVIDENCE  (label index 1)
Definition: The concept references observations, research, data, or prior work that
supports the claim. Evidence shows why the claim might be true.

Label EVIDENCE = 1 ONLY IF the text actually contains one or more of:
  - references to user research, interviews, or observations
  - data or statistics cited
  - prior art or existing solutions mentioned
  - specific findings that support the direction

Label EVIDENCE = 0 when the text shows any of these ABSENT signals:
  - claims without support
  - "I believe users want..." without demonstration
  - assumptions presented as facts
  - no reference to investigation or research

EDGE CASES (treat as 0):
  - PROSPECTIVE / PLANNED evidence is NOT evidence. "We will analyse the pilot data",
    "a study will be run", "we plan to interview users" → EVIDENCE = 0. The evidence
    does not exist yet.
  - Naming a method without a finding ("using surveys") with no result → 0.
  - DESCRIBING YOUR OWN PROGRAMME'S SCALE OR REACH is prior work, NOT evidence.
    "We work with 1,200 families across 25 villages", "our platform has 5,000 users",
    "we operate in 25 coastal villages" → these describe the deployment's size or scope,
    they do not support the claim. Such counts are EVIDENCE = 1 ONLY when paired with a
    reported OUTCOME or FINDING ("adoption rose 30%", "spoilage fell by half"). Beneficiary
    or scale numbers ALONE → EVIDENCE = 0.

================================================================================
DIMENSION: ASSUMPTIONS  (label index 3)
Definition: The concept EXPLICITLY names what must be true for the design to work —
the conditions, beliefs, or prerequisites that aren't proven.

Label ASSUMPTIONS = 1 ONLY IF the text explicitly states one or more of:
  - "We assume that..." / "This requires..." / "This depends on..."
  - a named dependency ("users must have smartphone access")
  - an acknowledged uncertainty
  - a stated condition for success

Label ASSUMPTIONS = 0 when the text shows any of these ABSENT signals:
  - hidden prerequisites
  - unstated dependencies
  - treating assumptions as facts
  - no acknowledgment of what could be wrong

CRITICAL: an assumption that YOU (the annotator) can infer but the TEXT does not
state is NOT acknowledged. Only an EXPLICITLY NAMED assumption counts. If the
author has not written it down, ASSUMPTIONS = 0.

BUT a stated DESIGN LIMITATION or CONSTRAINT counts as an acknowledged dependency →
ASSUMPTIONS = 1. Naming a boundary of what the design can do is acknowledging a
condition for success. Examples that ARE acknowledgement: "works best on laptops;
mobile functionality is limited", "requires reliable internet", "only available in
Hindi for now", "depends on users having bank accounts". If the author names what the
design needs or where it falls short, label 1.

================================================================================
Respond with ONLY a JSON object, no prose, no code fence:
{"EVIDENCE": 0 or 1, "EVIDENCE_reason": "<=15 words", "ASSUMPTIONS": 0 or 1, "ASSUMPTIONS_reason": "<=15 words"}"""

_thread_local = threading.local()


def get_client() -> openai.OpenAI:
    c = getattr(_thread_local, "client", None)
    if c is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            sys.exit("ERROR: OPENROUTER_API_KEY not set in environment.")
        c = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        _thread_local.client = c
    return c


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse(raw: str) -> dict:
    """Parse the model reply. Tolerates code fences and truncated JSON: the two
    integer labels precede the long reason strings, so a regex fallback recovers
    them even when the trailing reason is cut off."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1:
            s = s[i:j + 1]
    try:
        d = json.loads(s)
        return {
            "EVIDENCE": int(d["EVIDENCE"]),
            "EVIDENCE_reason": str(d.get("EVIDENCE_reason", ""))[:120],
            "ASSUMPTIONS": int(d["ASSUMPTIONS"]),
            "ASSUMPTIONS_reason": str(d.get("ASSUMPTIONS_reason", ""))[:120],
        }
    except Exception:
        ev = re.search(r'"EVIDENCE"\s*:\s*([01])', raw)
        asm = re.search(r'"ASSUMPTIONS"\s*:\s*([01])', raw)
        if not (ev and asm):
            raise ValueError(f"unparseable reply: {raw[:120]!r}")
        evr = re.search(r'"EVIDENCE_reason"\s*:\s*"([^"]*)"', raw)
        asr = re.search(r'"ASSUMPTIONS_reason"\s*:\s*"([^"]*)"', raw)
        return {
            "EVIDENCE": int(ev.group(1)),
            "EVIDENCE_reason": (evr.group(1) if evr else "[recovered]")[:120],
            "ASSUMPTIONS": int(asm.group(1)),
            "ASSUMPTIONS_reason": (asr.group(1) if asr else "[recovered]")[:120],
        }


def classify(text: str) -> dict:
    """Return {'EVIDENCE':0/1,'EVIDENCE_reason':str,'ASSUMPTIONS':0/1,'ASSUMPTIONS_reason':str}."""
    resp = get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Design-concept text:\n\n{text}"},
        ],
    )
    return _parse(resp.choices[0].message.content)


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.open():
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["key"]] = rec["result"]
    return cache


def main():
    ap = argparse.ArgumentParser(description="Strict-rubric relabel of EVIDENCE + ASSUMPTIONS.")
    ap.add_argument("--input", required=True, help="dict-form source jsonl (e.g. train.jsonl)")
    ap.add_argument("--out-prefix", required=True, help="output prefix (e.g. train.relabel)")
    ap.add_argument("--sample", type=int, default=0, help="relabel only the first N records (validation run)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    # Empty-text rows can't be classified or evaluated — exclude them from the corpus.
    n_empty = sum(1 for r in records if not r["text"].strip())
    if n_empty:
        print(f"excluded {n_empty} empty-text record(s) from {args.input}", flush=True)
        records = [r for r in records if r["text"].strip()]
    if args.sample:
        records = records[: args.sample]

    cache_path = Path(f"{args.out_prefix}.cache.jsonl")
    cache = load_cache(cache_path)
    cache_lock = threading.Lock()
    print(f"{len(records)} records | {len(cache)} already cached", flush=True)

    todo = [r for r in records if text_key(r["text"]) not in cache]

    def work(rec):
        k = text_key(rec["text"])
        result = classify(rec["text"])
        with cache_lock:
            with cache_path.open("a") as f:
                f.write(json.dumps({"key": k, "result": result}) + "\n")
            cache[k] = result
        return k

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # one bad record must not sink the batch
                r = futures[fut]
                print(f"  WARN failed on: {r['text'][:60]!r}: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} new records done", flush=True)

    # ---- assemble outputs ----
    dict_out = Path(f"{args.out_prefix}.dict.jsonl")
    deberta_out = Path(f"{args.out_prefix}.deberta.jsonl")
    diff_out = Path(f"{args.out_prefix}.diff.csv")

    ev_flips = {"1to0": 0, "0to1": 0}
    as_flips = {"1to0": 0, "0to1": 0}
    n_changed = 0

    with dict_out.open("w") as fd, deberta_out.open("w") as fb, diff_out.open("w", newline="") as fc:
        writer = csv.writer(fc)
        writer.writerow(["text", "old_EV", "new_EV", "EV_reason", "old_AS", "new_AS", "AS_reason", "changed"])
        for rec in records:
            k = text_key(rec["text"])
            res = cache.get(k)
            if res is None:
                continue  # skipped/failed — leave it out of corrected corpus
            old = rec["labels"]
            old_ev, old_as = int(old["EVIDENCE"]), int(old["ASSUMPTIONS"])
            new_ev, new_as = res["EVIDENCE"], res["ASSUMPTIONS"]

            if new_ev != old_ev:
                ev_flips["1to0" if old_ev == 1 else "0to1"] += 1
            if new_as != old_as:
                as_flips["1to0" if old_as == 1 else "0to1"] += 1
            changed = (new_ev != old_ev) or (new_as != old_as)
            n_changed += int(changed)

            new_labels = dict(old)
            new_labels["EVIDENCE"] = new_ev
            new_labels["ASSUMPTIONS"] = new_as

            # dict form (preserves original_tags if present)
            out_rec = dict(rec)
            out_rec["labels"] = new_labels
            fd.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            # list form for the trainer, in DIMENSION_ORDER
            fb.write(json.dumps({
                "text": rec["text"],
                "labels": [int(new_labels[d]) for d in DIMENSION_ORDER],
            }, ensure_ascii=False) + "\n")

            writer.writerow([
                rec["text"][:160], old_ev, new_ev, res["EVIDENCE_reason"],
                old_as, new_as, res["ASSUMPTIONS_reason"], "YES" if changed else "",
            ])

    n = sum(1 for rec in records if cache.get(text_key(rec["text"])) is not None)
    print("\n================ RELABEL SUMMARY ================")
    print(f"records relabelled : {n}/{len(records)}")
    print(f"EVIDENCE flips     : 1->0 {ev_flips['1to0']:5}   0->1 {ev_flips['0to1']:5}")
    print(f"ASSUMPTIONS flips  : 1->0 {as_flips['1to0']:5}   0->1 {as_flips['0to1']:5}")
    print(f"records changed    : {n_changed} ({(n_changed / n * 100) if n else 0:.0f}%)")
    print(f"\nwrote:\n  {dict_out}\n  {deberta_out}  <- drop-in for train_stage1.py\n  {diff_out}  <- SPOT-CHECK THIS BY HAND")
    print("=================================================")


if __name__ == "__main__":
    main()
