#!/usr/bin/env bash
# ============================================================================
# Coherence Diagnostic — consolidated retrain + honest-eval sequence
# Runs on the GPU machine. Covers Steps 3 + 4 of RELABEL-RUNBOOK.md.
# Emails me@prayas.in when finished (success OR failure).
#
# Prepared 1 June 2026. Run in YOUR terminal on the GPU box — NOT via an agent.
#
# What it does, in order:
#   0. Preflight  — checks corpus + scripts present, refuses if the manual
#                   relabel/spot-check gate (Steps 1-2) has not been done.
#   1. Environment — creates .train-venv, installs deps (idempotent).
#   2. CUDA check  — aborts if no CUDA (train_stage1.py hardcodes bf16=True; needs Ampere+).
#   3. Retrain     — train_stage1.py on the corrected labels -> model dir.
#   4. Honest eval — evaluate_model.py on OLD then NEW, same corrected holdout,
#                    both tee'd into one timestamped comparison log.
#   *. Notify      — emails the outcome to me@prayas.in, attaching the log.
#
# What it deliberately does NOT do:
#   - It does NOT relabel (Step 1: CPU + ~$4 OpenRouter, done on your laptop).
#   - It does NOT bypass the spot-check (Step 2: human gate on the diff CSV).
#   - It does NOT ship (Step 5: GitHub Release + deploy — explicit go-ahead only).
#
# --- Notification setup (set these on the GPU box before running) -----------
#   export POSTAL_SMTP_HOST=hammer.sovran.email
#   export POSTAL_SMTP_PORT=25
#   export POSTAL_SMTP_USERNAME=...        # Postal SMTP user
#   export POSTAL_SMTP_PASSWORD=...        # from CapRover env (Coherence demo app)
#   (If SMTP is not configured, the script still runs and just prints the outcome.)
# ============================================================================

set -euo pipefail

# --- Resolve to this script's own directory (the corpus folder) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Tunables (override on the command line, e.g. EPOCHS=8 ./run-...) --------
TRAIN_DATA="${TRAIN_DATA:-train.relabel.deberta.jsonl}"
VAL_DATA="${VAL_DATA:-val.relabel.deberta.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-model-v1.1-relabel}"
OLD_MODEL="${OLD_MODEL:-../models/deberta-coherence}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"
VENV="${VENV:-.train-venv}"

# --- Notification config (mirrors backend/error_notifier.py) ----------------
NOTIFY_EMAIL="${NOTIFY_EMAIL:-me@prayas.in}"
NOTIFY_FROM="${NOTIFY_FROM:-hello@koher.app}"
POSTAL_SMTP_HOST="${POSTAL_SMTP_HOST:-}"
POSTAL_SMTP_PORT="${POSTAL_SMTP_PORT:-25}"
POSTAL_SMTP_USERNAME="${POSTAL_SMTP_USERNAME:-}"
POSTAL_SMTP_PASSWORD="${POSTAL_SMTP_PASSWORD:-}"

# --- State for the exit-trap notifier ---------------------------------------
LAST_STEP="startup"
LOG=""
HOSTLABEL="$(hostname 2>/dev/null || echo gpu-box)"

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
ist() { TZ="Asia/Kolkata" date "+%d %B %Y, %H:%M IST"; }
say() { LAST_STEP="$1"; printf '\n\033[1m[%s] %s\033[0m\n' "$(ts)" "$1"; }

# Send the outcome email. Args: SUBJECT  BODY  [ATTACHMENT_PATH]
# Uses python3 stdlib (smtplib); never aborts the script if mail fails.
send_notification() {
  local subject="$1" body="$2" attach="${3:-}"
  if [[ -z "$POSTAL_SMTP_HOST" || -z "$POSTAL_SMTP_USERNAME" ]]; then
    echo "  [notify] SMTP not configured — skipping email. Outcome was:"
    echo "  [notify] $subject"
    return 0
  fi
  NOTIFY_SUBJECT="$subject" NOTIFY_BODY="$body" NOTIFY_ATTACH="$attach" \
  NOTIFY_EMAIL="$NOTIFY_EMAIL" NOTIFY_FROM="$NOTIFY_FROM" \
  POSTAL_SMTP_HOST="$POSTAL_SMTP_HOST" POSTAL_SMTP_PORT="$POSTAL_SMTP_PORT" \
  POSTAL_SMTP_USERNAME="$POSTAL_SMTP_USERNAME" POSTAL_SMTP_PASSWORD="$POSTAL_SMTP_PASSWORD" \
  python3 - <<'PY' || echo "  [notify] email send failed (see above) — outcome stands."
import os, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

msg = MIMEMultipart()
msg["From"] = os.environ["NOTIFY_FROM"]
msg["To"] = os.environ["NOTIFY_EMAIL"]
msg["Subject"] = os.environ["NOTIFY_SUBJECT"]
msg.attach(MIMEText(os.environ["NOTIFY_BODY"], "plain"))

attach = os.environ.get("NOTIFY_ATTACH", "")
if attach and os.path.isfile(attach):
    with open(attach, "rb") as fh:
        part = MIMEApplication(fh.read(), Name=os.path.basename(attach))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attach)}"'
    msg.attach(part)

host = os.environ["POSTAL_SMTP_HOST"]
port = int(os.environ.get("POSTAL_SMTP_PORT", "25"))
with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls()
    s.login(os.environ["POSTAL_SMTP_USERNAME"], os.environ["POSTAL_SMTP_PASSWORD"])
    s.sendmail(msg["From"], [msg["To"]], msg.as_string())
print(f"  [notify] emailed {msg['To']}: {msg['Subject']}")
PY
}

# Exit trap: fires on success AND failure. Captures rc FIRST.
on_exit() {
  local rc=$?
  local subject body
  if [[ "$rc" -eq 0 ]]; then
    subject="[Coherence retrain] DONE on ${HOSTLABEL} — $(ist)"
    body="Coherence retrain + honest-eval sequence finished OK.

Finished: $(ist)
Machine:  ${HOSTLABEL}
Folder:   ${SCRIPT_DIR}

New model:      ${SCRIPT_DIR}/${OUTPUT_DIR}
Comparison log: ${SCRIPT_DIR}/${LOG}   (attached)

What success looks like (NEW vs OLD in the attached log):
  - EVIDENCE and ASSUMPTIONS accuracy: UP
  - 'pred-PRESENT-when-ABSENT' for EVIDENCE/ASSUMPTIONS: DOWN sharply  (Pablo's bug)
  - Probe 'fluent, NO evidence': EV and AS drop below 0.50
  - Probes 'WITH real evidence' / 'EXPLICIT assumption': stay high

NEXT (Step 5 — SHIP) is not automated and needs an explicit 'ship' go-ahead.
See RELABEL-RUNBOOK.md, Step 5."
  else
    subject="[Coherence retrain] FAILED at '${LAST_STEP}' on ${HOSTLABEL} — $(ist)"
    body="Coherence retrain sequence FAILED.

Failed at step: ${LAST_STEP}
Exit code:      ${rc}
Time:           $(ist)
Machine:        ${HOSTLABEL}
Folder:         ${SCRIPT_DIR}

Re-run after fixing. Scroll the terminal for the error above the trap.
(No model was shipped; the deployed v1.0 model is untouched.)"
  fi
  send_notification "$subject" "$body" "${LOG:-}"
  exit "$rc"
}
trap on_exit EXIT

# ----------------------------------------------------------------------------
say "STEP 0 — Preflight"
# ----------------------------------------------------------------------------
missing=0
for f in train_stage1.py evaluate_model.py "$TRAIN_DATA" "$VAL_DATA"; do
  if [[ ! -e "$f" ]]; then
    echo "  MISSING: $f"
    missing=1
  fi
done

if [[ "$missing" -eq 1 ]]; then
  cat <<'EOF'

  The relabelled corpus and/or training scripts are not all present.

  The relabel + spot-check (Steps 1-2 of RELABEL-RUNBOOK.md) happen on your
  laptop BEFORE this GPU sequence, and are intentionally NOT automated here:

      export OPENROUTER_API_KEY=...
      python relabel_to_rubric.py --input train.jsonl --out-prefix train.relabel --concurrency 10
      python relabel_to_rubric.py --input val.jsonl   --out-prefix val.relabel   --concurrency 10
      # then eyeball train.relabel.diff.csv (the human gate), THEN sync to the GPU box.

  Aborting. Nothing was changed.
EOF
  exit 1
fi

train_rows=$(wc -l < "$TRAIN_DATA" | tr -d ' ')
val_rows=$(wc -l < "$VAL_DATA" | tr -d ' ')
echo "  train: $TRAIN_DATA ($train_rows rows)"
echo "  val:   $VAL_DATA ($val_rows rows)   <- held out, NOT used in training"
echo "  old model (for before/after): $OLD_MODEL"
if [[ ! -e "$OLD_MODEL/model.safetensors" && ! -e "$OLD_MODEL/pytorch_model.bin" ]]; then
  echo "  WARNING: old model weights not found at $OLD_MODEL — the 'before' half"
  echo "           of the comparison will be skipped. (Training still proceeds.)"
fi

# ----------------------------------------------------------------------------
say "STEP 1 — Environment ($VENV)"
# ----------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip
python -m pip install -q -U torch transformers sentencepiece protobuf tiktoken scikit-learn numpy accelerate
echo "  deps ready: $(python -c 'import torch,transformers; print("torch",torch.__version__,"| transformers",transformers.__version__)')"

# ----------------------------------------------------------------------------
say "STEP 2 — CUDA check"
# ----------------------------------------------------------------------------
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  cat <<'EOF'

  No CUDA GPU detected. train_stage1.py sets bf16=True (Ampere+ tensor cores)
  and will error on CPU/MPS. This script is meant for the GPU box. If you really
  mean to train on CPU/MPS, edit train_stage1.py (bf16=True -> bf16=False) and
  re-run. Aborting.
EOF
  exit 1
fi
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ----------------------------------------------------------------------------
say "STEP 3 — Retrain on corrected labels -> $OUTPUT_DIR"
# ----------------------------------------------------------------------------
echo "  (Ignore train_stage1.py's final accuracy line — that is fit-to-its-own-labels,"
echo "   NOT a before/after. The real measurement is Step 4 below.)"
python train_stage1.py \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LR"

# ----------------------------------------------------------------------------
say "STEP 4 — Honest before/after on the SAME corrected holdout"
# ----------------------------------------------------------------------------
LOG="eval-compare-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "Coherence retrain before/after — generated $(ts)"
  echo "Holdout: $VAL_DATA ($val_rows rows). Training data: $TRAIN_DATA ($train_rows rows)."
  echo
  echo "=================== OLD MODEL ($OLD_MODEL) ==================="
} | tee "$LOG"

if [[ -e "$OLD_MODEL/model.safetensors" || -e "$OLD_MODEL/pytorch_model.bin" ]]; then
  python evaluate_model.py --model "$OLD_MODEL" --data "$VAL_DATA" 2>&1 | tee -a "$LOG"
else
  echo "  (skipped — old model weights not found at $OLD_MODEL)" | tee -a "$LOG"
fi

{
  echo
  echo "=================== NEW MODEL ($OUTPUT_DIR) ==================="
} | tee -a "$LOG"
python evaluate_model.py --model "$OUTPUT_DIR" --data "$VAL_DATA" 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
say "DONE — sequence complete"
# ----------------------------------------------------------------------------
cat <<EOF

  New model:        $SCRIPT_DIR/$OUTPUT_DIR
  Comparison log:   $SCRIPT_DIR/$LOG   <- this side-by-side IS the evidence.

  What success looks like (NEW vs OLD in the log):
    - EVIDENCE and ASSUMPTIONS accuracy: UP
    - "pred-PRESENT-when-ABSENT" for EVIDENCE/ASSUMPTIONS: DOWN sharply  (Pablo's bug)
    - Probe "fluent, NO evidence": EV and AS drop below 0.50
    - Probes "WITH real evidence" / "EXPLICIT assumption": stay high

  An email is on its way to $NOTIFY_EMAIL with the log attached.

  NEXT (Step 5 — SHIP) is NOT automated and needs an explicit "ship" go-ahead.
  See RELABEL-RUNBOOK.md "Step 5 — Ship": tar -> GitHub Release v1.1 ->
  bump entrypoint.sh URL -> clear /app/models volume -> redeploy -> re-test live.
EOF
# on_exit trap fires here and sends the success email.
