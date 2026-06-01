# Coherence EVIDENCE/ASSUMPTIONS relabel + retrain — runbook

**Created:** 27 May 2026, 20:49
**Why:** Pablo Martínez (Fellowship cohort) found that a fluent, evidence-free concept
scored EVIDENCE 98% and ASSUMPTIONS 100%. Audit of the training corpus confirmed the
labels drifted looser than the tool's own rubric (`spec.md`):

- 64% of `EVIDENCE=1` examples reference no observation/research/data/prior work.
- 71% of `ASSUMPTIONS=1` examples name no explicit dependency or condition.

The DeBERTa model reproduced that drift faithfully. The fix is upstream: re-derive
**only** the EVIDENCE and ASSUMPTIONS labels by applying the rubric strictly, then
retrain. CLAIM, SCOPE, GAPS are untouched.

This is the *current-architecture* (738 MB `deberta-v3-base`) fix, kept separate from
the v2 rebuild (≤100 MB, visual input, UI). The corrected corpus is the durable asset —
it feeds v2 later regardless; the one interim retrain is cheap.

---

## The honest framing (read this before trusting any number)

`train_stage1.py` reports accuracy **against the labels it trains on**. Retrain on
corrected labels and it will *also* report ~98% — the model fits whatever labels it is
given. **That is not a before/after.** The real measurement is `evaluate_model.py`:
score the OLD and NEW models against the SAME corrected test set (`val.relabel`) and
the fixed probes. The number that matters is EVIDENCE/ASSUMPTIONS accuracy on corrected
labels, and the "predicted PRESENT when label says ABSENT" over-prediction rate — that
over-prediction *is* Pablo's bug, and it should fall sharply.

---

## Files in this folder

| File | What it is |
|------|-----------|
| `relabel_to_rubric.py` | Strict-rubric relabeller (EVIDENCE + ASSUMPTIONS only). LLM = Claude Haiku 4.5, temp 0. Idempotent (per-text cache), resumable, emits a diff CSV. |
| `evaluate_model.py` | Honest before/after eval. Scores any model dir against a corrected test set + fixed probes. |
| `train_stage1.py` | Existing trainer (`deberta-v3-base`, multi-label, self-splits 90/10). GPU. |
| `train.relabel.*` | Relabel outputs (see Step 1). |
| `val.relabel.*` | Relabel outputs for the held-out val set — the clean test set. |

---

## Step 1 — Relabel (no GPU; ~$4; in progress / done)

```bash
export OPENROUTER_API_KEY=...        # from tools-demo/0a-coherence-diagnostic/env.txt
python relabel_to_rubric.py --input train.jsonl --out-prefix train.relabel --concurrency 10
python relabel_to_rubric.py --input val.jsonl   --out-prefix val.relabel   --concurrency 10
```

Outputs per prefix:
- `*.deberta.jsonl` — list-form `{text, labels:[CLAIM,EVIDENCE,SCOPE,ASSUMPTIONS,GAPS]}`, **drop-in for the trainer**.
- `*.dict.jsonl` — dict-form, preserves `original_tags`.
- `*.diff.csv` — every flip with the model's one-line reason. **The spot-check artefact.**
- `*.cache.jsonl` — per-text cache; safe to re-run, skips done work.

Two rubric edges were adjudicated (Prayas, 27 May 2026) and baked into the prompt:
- Own-programme **scale/reach** ("works with 1,200 families") = prior work, **not** evidence → EVIDENCE 0 unless an outcome/finding is reported.
- A stated **design limitation/constraint** ("works best on laptops; mobile limited") **is** an acknowledged dependency → ASSUMPTIONS 1.

## Step 2 — Spot-check (HUMAN GATE — this is what makes it a real fix)

Open `train.relabel.diff.csv`. Skim the flips. Pay most attention to:
- **`0→1` flips** (label gained) — these are the riskier direction; confirm each truly meets the rubric.
- A sample of `1→0` flips — confirm the stripped label really lacked evidence / an explicit assumption.

If a systematic miscall appears, adjust the rubric text in `relabel_to_rubric.py`
(SYSTEM_PROMPT), delete the affected `*.cache.jsonl`, and re-run. Do not proceed to
retrain on a corpus you have not eyeballed.

## Step 3 — Retrain (GPU — run in a terminal, never via an agent)

Train on the corrected train set; keep `val.relabel.deberta.jsonl` untouched as the test set.

```bash
python train_stage1.py \
  --train_data train.relabel.deberta.jsonl \
  --output_dir model-v1.1-relabel \
  --num_epochs 5 --batch_size 8 --learning_rate 2e-5
```

(`--model_name` defaults to `microsoft/deberta-v3-base`. Ignore the script's final
accuracy line for before/after — see "honest framing" above.)

## Step 4 — Honest before/after (no GPU needed; CPU fine on 723 rows)

```bash
# OLD (currently deployed) model
python evaluate_model.py --model ../models/deberta-coherence --data val.relabel.deberta.jsonl
# NEW model
python evaluate_model.py --model model-v1.1-relabel          --data val.relabel.deberta.jsonl
```

Success looks like: NEW model's EVIDENCE/ASSUMPTIONS **accuracy up** and
**"pred-PRESENT-when-ABSENT" down**, and on the probes, the fluent-no-evidence probe's
`EV`/`AS` drop below 0.5 while the real-evidence / explicit-assumption probes stay high.
Keep this output — it is the evidence for Pablo, and usable for grant impact reporting.

## Step 5 — Ship (REQUIRES EXPLICIT "ship" GO-AHEAD — release-artefact rule)

The deployed model is pulled by `entrypoint.sh` from a **GitHub Release**:
`coherence-diagnostic/releases/download/v1.0/model.tar.gz`. To ship the retrained model:

1. `tar -czf model.tar.gz -C model-v1.1-relabel .`
2. Create release **v1.1** on `koherarchitecture/coherence-diagnostic`, attach `model.tar.gz`.
3. Bump the URL in `entrypoint.sh` (both demo and release trees) to `…/v1.1/model.tar.gz`.
4. The persistent `/app/models` volume already holds v1.0 and won't re-download
   (entrypoint checks `if [ ! -f model.safetensors ]`). **Clear it** (or replace the
   file in-volume) so v1.1 is fetched. Then redeploy:
   `caprover deploy -n myplaceholder -a demo -t coherence-diagnostic.tar`
5. Re-run Pablo's exact input on `coherence-demo.koher.app` and confirm the score moved.

> Per the binding rule, do not upload the release artefact or push without an explicit
> "ship"/"release" instruction. Steps 1–4 are staged; Prayas triggers the upload.

---

## What this fixes — and what it does NOT

**Fixes:** the permissive, fluency-rewarding labels, and therefore the model's
over-scoring of evidence/assumptions. Directly answers Pablo's catch.

**Does not reach:** the structural ceiling. Even a perfectly relabelled model is still a
classifier scoring "present"; it cannot *abstain*, cannot natively detect absence, and
cannot name *which* implicit assumption matters. That residue is v2 / Tier-3 territory
(an absence/grounding stage — the essay's "abstain by design when it has no archive").
Do not let the relabel be reported as having solved the architectural point.
