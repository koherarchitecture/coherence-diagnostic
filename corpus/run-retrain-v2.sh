#!/usr/bin/env bash
# ============================================================================
# Coherence Diagnostic — retrain v2: INSTRUMENTED + GATED
# Successor to run-retrain-sequence.sh after the 1 June 2026 silent collapse.
# Run on the GPU box — NOT via an agent.
#
# What changed vs v1 (and WHY):
#   * CANARY FIRST — overfit ~16 rows (fp32, ~20s). If the stack can't learn that,
#     it can't learn anything; we abort BEFORE the 30-min run. (1 June burned a full
#     run to discover a non-learning stack.)
#   * fp32 by default — the 1 June run used bf16 and never learned. fp32 is the first
#     hypothesis; --bf16 / --fp16 are available to compare.
#   * HONEST GATE — the email says PASS / FAILED-COLLAPSE / FAILED-WEIGHTS /
#     INCONCLUSIVE based on real metrics. A collapsed model can no longer send "DONE".
#   * SAVE EVERYTHING — a diagnostics-<ts>/ bundle (env, versions, pip freeze,
#     nvidia-smi, full train log, per-epoch metrics, weight check, before/after eval
#     JSON+txt, verdict) is written and emailed as a tarball. This is the data we
#     lacked on 1 June.
#
# SMTP notify: source ./.smtp.env first (gitignored). If unset, it just prints.
# ============================================================================

set -uo pipefail   # deliberately NOT -e: collect diagnostics even when a step fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Tunables ---------------------------------------------------------------
TRAIN_DATA="${TRAIN_DATA:-train.relabel.deberta.jsonl}"
VAL_DATA="${VAL_DATA:-val.relabel.deberta.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-model-v1.1-relabel}"
OLD_MODEL="${OLD_MODEL:-../models/deberta-coherence}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"
VENV="${VENV:-.train-venv}"
PRECISION="${PRECISION:-fp32}"          # fp32 | bf16 | fp16
SKIP_CANARY="${SKIP_CANARY:-0}"         # set 1 only if you already proved the stack learns

PREC_FLAG=""
[[ "$PRECISION" == "bf16" ]] && PREC_FLAG="--bf16"
[[ "$PRECISION" == "fp16" ]] && PREC_FLAG="--fp16"

# --- Notify config ----------------------------------------------------------
NOTIFY_EMAIL="${NOTIFY_EMAIL:-me@prayas.in}"
NOTIFY_FROM="${NOTIFY_FROM:-hello@koher.app}"
POSTAL_SMTP_HOST="${POSTAL_SMTP_HOST:-}"
POSTAL_SMTP_PORT="${POSTAL_SMTP_PORT:-25}"
POSTAL_SMTP_USERNAME="${POSTAL_SMTP_USERNAME:-}"
POSTAL_SMTP_PASSWORD="${POSTAL_SMTP_PASSWORD:-}"

HOSTLABEL="$(hostname 2>/dev/null || echo gpu-box)"
LAST_STEP="startup"
DIAG="$SCRIPT_DIR/diagnostics-$(date +%Y%m%d-%H%M%S)"
VERDICT_FILE="$DIAG/VERDICT.txt"
mkdir -p "$DIAG"

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
ist() { TZ="Asia/Kolkata" date "+%d %B %Y, %H:%M IST"; }
say() { LAST_STEP="$1"; printf '\n\033[1m[%s] %s\033[0m\n' "$(ts)" "$1"; }

send_notification() {
  local subject="$1" body="$2" attach="${3:-}"
  if [[ -z "$POSTAL_SMTP_HOST" || -z "$POSTAL_SMTP_USERNAME" ]]; then
    echo "  [notify] SMTP not configured — skipping email. Outcome was:"; echo "  [notify] $subject"
    return 0
  fi
  NOTIFY_SUBJECT="$subject" NOTIFY_BODY="$body" NOTIFY_ATTACH="$attach" \
  NOTIFY_EMAIL="$NOTIFY_EMAIL" NOTIFY_FROM="$NOTIFY_FROM" \
  POSTAL_SMTP_HOST="$POSTAL_SMTP_HOST" POSTAL_SMTP_PORT="$POSTAL_SMTP_PORT" \
  POSTAL_SMTP_USERNAME="$POSTAL_SMTP_USERNAME" POSTAL_SMTP_PASSWORD="$POSTAL_SMTP_PASSWORD" \
  python3 - <<'PY' || echo "  [notify] email send failed — outcome stands (see bundle)."
import os, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
msg = MIMEMultipart()
msg["From"] = os.environ["NOTIFY_FROM"]; msg["To"] = os.environ["NOTIFY_EMAIL"]
msg["Subject"] = os.environ["NOTIFY_SUBJECT"]
msg.attach(MIMEText(os.environ["NOTIFY_BODY"], "plain"))
attach = os.environ.get("NOTIFY_ATTACH", "")
if attach and os.path.isfile(attach):
    with open(attach, "rb") as fh:
        part = MIMEApplication(fh.read(), Name=os.path.basename(attach))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attach)}"'
    msg.attach(part)
host = os.environ["POSTAL_SMTP_HOST"]; port = int(os.environ.get("POSTAL_SMTP_PORT", "25"))
with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls(); s.login(os.environ["POSTAL_SMTP_USERNAME"], os.environ["POSTAL_SMTP_PASSWORD"])
    s.sendmail(msg["From"], [msg["To"]], msg.as_string())
print(f"  [notify] emailed {msg['To']}: {msg['Subject']}")
PY
}

on_exit() {
  local rc=$?
  local tarball="${DIAG}.tar.gz" subject body
  tar -czf "$tarball" -C "$SCRIPT_DIR" "$(basename "$DIAG")" 2>/dev/null || tarball=""
  if [[ -f "$VERDICT_FILE" ]]; then
    local tag; tag="$(head -1 "$VERDICT_FILE")"
    subject="[Coherence retrain v2] ${tag} on ${HOSTLABEL} — $(ist)"
    body="$(cat "$VERDICT_FILE")

Machine:        ${HOSTLABEL}
Folder:         ${SCRIPT_DIR}
Diagnostics:    ${DIAG}  (attached as $(basename "${tarball:-none}"))
New model dir:  ${SCRIPT_DIR}/${OUTPUT_DIR}

Reminder: the deployed v1.0 model is untouched. Step 5 (SHIP) is manual and needs an
explicit 'ship' go-ahead, and only if the verdict is PASS and you accept the before/after."
  else
    subject="[Coherence retrain v2] ABORTED (rc=${rc}) at '${LAST_STEP}' on ${HOSTLABEL} — $(ist)"
    body="The v2 sequence aborted before writing a verdict.

Aborted at:  ${LAST_STEP}
Exit code:   ${rc}
Time:        $(ist)
Machine:     ${HOSTLABEL}
Diagnostics: ${DIAG}  (attached — read the latest .log for the error)
(No model shipped; deployed v1.0 untouched.)"
  fi
  send_notification "$subject" "$body" "${tarball:-}"
  exit "$rc"
}
trap on_exit EXIT

# ----------------------------------------------------------------------------
say "STEP 0 — Preflight + environment capture -> $DIAG/env.txt"
# ----------------------------------------------------------------------------
missing=0
for f in train_stage1.py canary_overfit.py evaluate_model.py "$TRAIN_DATA" "$VAL_DATA"; do
  [[ -e "$f" ]] || { echo "  MISSING: $f"; missing=1; }
done
[[ "$missing" -eq 1 ]] && { echo "Aborting: required files missing."; exit 1; }

train_rows=$(wc -l < "$TRAIN_DATA" | tr -d ' ')
val_rows=$(wc -l < "$VAL_DATA" | tr -d ' ')

# venv + deps (tee install log)
if [[ ! -d "$VENV" ]]; then python3 -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip
# Pin transformers to the version that trained the known-good 98% model (recorded in
# ../models/deberta-coherence/config.json: transformers_version 4.57.1). The 1-June v2
# collapse ran on transformers 5.x (major jump) and NaN'd in fp32 — precision-independent,
# so the stack is the suspect. Override with TRANSFORMERS_PIN= to test other versions.
python -m pip install -q -U torch "transformers==${TRANSFORMERS_PIN:-4.57.1}" sentencepiece protobuf tiktoken scikit-learn numpy accelerate 2>&1 | tee "$DIAG/install.log"

{
  echo "Coherence retrain v2 — environment  ($(ts))"
  echo "host: $HOSTLABEL"
  echo "precision requested: $PRECISION   epochs: $EPOCHS  batch: $BATCH_SIZE  lr: $LR"
  echo "train: $TRAIN_DATA ($train_rows rows)   val(holdout): $VAL_DATA ($val_rows rows)"
  echo "old model: $OLD_MODEL"
  echo "--- git ---"; git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo "(not a git checkout)"
  echo "--- uname ---"; uname -a
  echo "--- python / torch / transformers / cuda ---"
  python -c 'import sys,torch,transformers; print("python",sys.version.split()[0]); print("torch",torch.__version__,"cuda",torch.version.cuda,"avail",torch.cuda.is_available()); print("transformers",transformers.__version__); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
  echo "--- nvidia-smi ---"; command -v nvidia-smi >/dev/null && nvidia-smi || echo "(no nvidia-smi)"
  echo "--- pip freeze ---"; python -m pip freeze
} > "$DIAG/env.txt" 2>&1
echo "  env captured. train=$train_rows val=$val_rows"

# CUDA check (warn, don't hard-fail — fp32 CPU still trains, just slow)
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "  WARNING: no CUDA GPU. fp32 will run on CPU (slow). bf16/fp16 would error."
fi

# ----------------------------------------------------------------------------
say "STEP 1 — CANARY (overfit ~16 rows, $PRECISION) — can this stack learn at all?"
# ----------------------------------------------------------------------------
if [[ "$SKIP_CANARY" == "1" ]]; then
  echo "  SKIP_CANARY=1 — skipping (you asserted the stack already learns)."
else
  python canary_overfit.py --train_data "$TRAIN_DATA" $PREC_FLAG 2>&1 | tee "$DIAG/canary.log"
  canary_rc=${PIPESTATUS[0]}
  if [[ "$canary_rc" -ne 0 ]]; then
    {
      echo "FAILED-MACHINERY"
      echo
      echo "CANARY FAILED — the stack could not overfit 16 examples in $PRECISION."
      echo "This is NOT a data problem (the OLD model scores ~0.92 on this same holdout)"
      echo "and NOT the relabel. It is the training stack: precision kernels, a"
      echo "library/CUDA mismatch in the freshly-built venv, or tokenisation."
      echo
      echo "DO NOT launch the full run — it cannot succeed while the canary fails."
      echo "Next moves (in order):"
      echo "  1. Try another precision:  PRECISION=bf16 ./run-retrain-v2.sh   (and fp16)"
      echo "  2. Pin the January stack (the only thing new vs the 98% run is this"
      echo "     latest-everything venv + CUDA). Recreate .train-venv with the versions"
      echo "     from a known-good run and re-canary."
      echo "  3. Data-vs-stack proof, if ever in doubt: train on OLD train.jsonl with"
      echo "     this venv — collapse there too = 100% stack."
      echo
      echo "See $DIAG/canary.log and $DIAG/env.txt (exact versions to pin)."
    } > "$VERDICT_FILE"
    echo "  CANARY FAILED — see verdict. Aborting before the full run."
    exit 3
  fi
  echo "  CANARY PASSED — the stack can learn. Proceeding to the full run."
fi

# ----------------------------------------------------------------------------
say "STEP 2 — Full retrain ($PRECISION) -> $OUTPUT_DIR   (full log -> train.log)"
# ----------------------------------------------------------------------------
python train_stage1.py \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LR" \
  $PREC_FLAG 2>&1 | tee "$DIAG/train.log"
train_rc=${PIPESTATUS[0]}
cp -f "$OUTPUT_DIR/train-metrics.json" "$DIAG/" 2>/dev/null || true
ls -la "$OUTPUT_DIR" > "$DIAG/model-dir-listing.txt" 2>&1
find "$OUTPUT_DIR" -name "*.safetensors" -o -name "pytorch_model*.bin" >> "$DIAG/model-dir-listing.txt" 2>&1
echo "  train exited rc=$train_rc (3 = collapsed/weights-bad per train_stage1.py VERDICT)."

# ----------------------------------------------------------------------------
say "STEP 3 — Honest before/after on the SAME corrected holdout (JSON + txt)"
# ----------------------------------------------------------------------------
COMPARE="$DIAG/eval-compare.txt"
{
  echo "Coherence retrain v2 before/after — $(ts)"
  echo "Holdout: $VAL_DATA ($val_rows rows).  precision=$PRECISION epochs=$EPOCHS lr=$LR"
  echo; echo "=================== OLD MODEL ($OLD_MODEL) ==================="
} | tee "$COMPARE"
if [[ -e "$OLD_MODEL/model.safetensors" || -e "$OLD_MODEL/pytorch_model.bin" ]]; then
  python evaluate_model.py --model "$OLD_MODEL" --data "$VAL_DATA" --json "$DIAG/eval-old.json" 2>&1 | tee -a "$COMPARE"
else
  echo "  (skipped — old model weights not found at $OLD_MODEL)" | tee -a "$COMPARE"
fi
{ echo; echo "=================== NEW MODEL ($OUTPUT_DIR) ==================="; } | tee -a "$COMPARE"
python evaluate_model.py --model "$OUTPUT_DIR" --data "$VAL_DATA" --json "$DIAG/eval-new.json" 2>&1 | tee -a "$COMPARE"
new_eval_rc=${PIPESTATUS[0]}

# ----------------------------------------------------------------------------
say "STEP 4 — Verdict (PASS / FAILED-COLLAPSE / FAILED-WEIGHTS / INCONCLUSIVE)"
# ----------------------------------------------------------------------------
TRAIN_RC="$train_rc" NEW_EVAL_RC="$new_eval_rc" \
OLD_JSON="$DIAG/eval-old.json" NEW_JSON="$DIAG/eval-new.json" \
PRECISION="$PRECISION" EPOCHS="$EPOCHS" LR="$LR" \
python - <<'PY' > "$VERDICT_FILE" 2>&1
import json, os
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
new = load(os.environ["NEW_JSON"]); old = load(os.environ["OLD_JSON"])
train_rc = int(os.environ.get("TRAIN_RC", "1"))
prec = os.environ["PRECISION"]; ep = os.environ["EPOCHS"]; lr = os.environ["LR"]

def od(m, dim): return m["per_dim"][dim]["over_predict_when_absent"]

lines, tag = [], "INCONCLUSIVE"
if new is None:
    tag = "FAILED-WEIGHTS"
    lines += ["NEW model could not be evaluated — no reloadable metrics (weights likely",
              "missing or non-finite). train exit rc=%d. See train.log / model-dir-listing.txt." % train_rc]
elif new.get("collapsed") or new.get("f1_micro", 0) == 0.0 or new.get("pred_pos_frac", 0) < 1e-6:
    tag = "FAILED-COLLAPSE"
    lines += ["NEW model COLLAPSED to all-negative (f1_micro=%.3f, pred_pos_frac=%.3f) under"
              % (new.get("f1_micro", 0), new.get("pred_pos_frac", 0)),
              "precision=%s lr=%s epochs=%s." % (prec, lr, ep),
              "But the CANARY passed (the stack CAN learn), so this is optimisation dynamics,",
              "not the stack: try a different precision or lower LR / longer warmup.",
              "If the canary was skipped, run it — it may BE the stack."]
elif train_rc != 0:
    tag = "FAILED-WEIGHTS"
    lines += ["NEW model evaluates but train_stage1 exited rc=%d (weights/save problem)." % train_rc,
              "Inspect train.log [save] lines and model-dir-listing.txt before trusting it."]
else:
    # Model learned and saved. Did it fix Pablo's bug vs OLD?
    f1 = new.get("f1_micro", 0)
    if old is None:
        tag = "TRAINED-NO-BASELINE"
        lines += ["NEW model learned (f1_micro=%.3f) but OLD baseline missing — cannot judge the fix." % f1]
    else:
        ev_old, ev_new = od(old, "EVIDENCE"), od(new, "EVIDENCE")
        as_old, as_new = od(old, "ASSUMPTIONS"), od(new, "ASSUMPTIONS")
        # probe: Pablo-shape (index 0) EV should drop <0.5; real-evidence (index1) EV stays high
        pr = {p["label"][:24]: p for p in new.get("probes", [])}
        pablo = next((p for p in new["probes"] if "Pablo-shape" in p["label"]), None)
        realev = next((p for p in new["probes"] if "WITH real evidence" in p["label"]), None)
        improved = (ev_new < ev_old) and (as_new < as_old)
        probes_ok = (pablo and realev and pablo["EV"] < 0.5 and realev["EV"] >= 0.5)
        if improved and probes_ok:
            tag = "PASS"
            lines += ["NEW model learned AND reduced Pablo's over-prediction:",
                      "  EVIDENCE  pred-present-when-absent: %.3f -> %.3f" % (ev_old, ev_new),
                      "  ASSUMPTIONS pred-present-when-absent: %.3f -> %.3f" % (as_old, as_new),
                      "  probe 'Pablo-shape' EV=%.2f (<0.5 good); 'real evidence' EV=%.2f (>=0.5 good)"
                      % (pablo["EV"], realev["EV"]),
                      "f1_micro=%.3f, mean_acc=%.3f. Looks ship-worthy — YOUR call on the before/after."
                      % (f1, new.get("mean_acc", 0))]
        else:
            tag = "INCONCLUSIVE"
            lines += ["NEW model learned (f1_micro=%.3f) but the bug-fix signal is NOT clean:" % f1,
                      "  EVIDENCE over-predict: %.3f -> %.3f  (want DOWN)" % (ev_old, ev_new),
                      "  ASSUMPTIONS over-predict: %.3f -> %.3f  (want DOWN)" % (as_old, as_new),
                      "  Pablo-shape probe EV=%.2f / real-evidence probe EV=%.2f (want <0.5 / >=0.5)"
                      % (pablo["EV"] if pablo else -1, realev["EV"] if realev else -1),
                      "This is now a real result about the FIX (not the stack): the relabel",
                      "may need revisiting, or more epochs. Read eval-compare.txt."]
print(tag); print()
print("\n".join(lines))
print(); print("Full evidence: eval-compare.txt, eval-old.json, eval-new.json, train-metrics.json, env.txt")
PY

echo
echo "============================================================"
cat "$VERDICT_FILE"
echo "============================================================"
echo "Bundle: $DIAG  (emailed as a tarball to $NOTIFY_EMAIL)"

# exit code mirrors the verdict tag (0 only on PASS); trap emails using VERDICT_FILE.
if head -1 "$VERDICT_FILE" | grep -q '^PASS'; then exit 0; else exit 2; fi
