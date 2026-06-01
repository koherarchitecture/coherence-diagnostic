import argparse
import json
import os
import platform
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class CoherenceDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        with open(path) as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["labels"], dtype=torch.float)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()

    # Per-dimension accuracy
    accuracies = {}
    dimensions = ["CLAIM", "EVIDENCE", "SCOPE", "ASSUMPTIONS", "GAPS"]
    for i, dim in enumerate(dimensions):
        accuracies[f"accuracy_{dim}"] = accuracy_score(labels[:, i], predictions[:, i])

    # Overall metrics
    accuracies["accuracy_mean"] = np.mean([v for k, v in accuracies.items()])
    accuracies["f1_micro"] = f1_score(labels, predictions, average="micro")
    # Collapse telltale: fraction of ALL predictions that are positive. ~0.0 means the
    # model has collapsed to predicting "absent" for everything (the 1 June failure mode).
    accuracies["pred_pos_frac"] = float(predictions.mean())

    return accuracies

def main(args):
    # --- Environment banner (captured in the run log; this is the data we lacked
    #     after the 1 June collapse — exact stack + precision actually used). ------
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
    print("=" * 70)
    print(f"[env] python {platform.python_version()} | torch {torch.__version__} "
          f"| transformers {transformers.__version__}")
    print(f"[env] cuda_available={torch.cuda.is_available()} | device={dev} "
          f"| cuda={torch.version.cuda}")
    print(f"[run] precision={precision} | lr={args.learning_rate} "
          f"| warmup_ratio={args.warmup_ratio} | max_grad_norm={args.max_grad_norm} "
          f"| weight_decay={args.weight_decay} | epochs={args.num_epochs} "
          f"| batch={args.batch_size}")
    print(f"[run] best-model metric = f1_micro (NOT accuracy_mean — that rewards the "
          f"all-negative collapse)")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,
        problem_type="multi_label_classification"
    )

    train_dataset = CoherenceDataset(args.train_data, tokenizer)

    # Split for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # Precision is now a flag (default fp32). The 1 June run used bf16 and the
        # model never learned; fp32 is the first thing the canary/full run tests.
        bf16=args.bf16,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # f1_micro, NOT accuracy_mean: a collapsed all-negative model has the SAME
        # accuracy_mean (= base rate) every epoch, so accuracy_mean cannot select a
        # good epoch and silently blesses the collapse. f1_micro=0 exposes it.
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        logging_first_step=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Persist the FULL per-epoch history (loss / acc / f1 / pred_pos_frac). This
    #     is the file we wished we had after 1 June. ------------------------------
    metrics_path = os.path.join(args.output_dir, "train-metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"\n[save] per-epoch history -> {metrics_path}")

    # --- Verify weights actually landed, with a NaN/inf scan. The 1 June model dir
    #     had NO weight file; never declare success on a dir we cannot reload. -----
    def _weight_file(d):
        for n in ("model.safetensors", "pytorch_model.bin"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        return None

    wf = _weight_file(args.output_dir)
    if wf is None:
        # safetensors can silently refuse DeBERTa-v3's tied weights — force a torch save.
        print("[save] WARNING: no weight file after save_model; forcing safe_serialization=False")
        trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
        wf = _weight_file(args.output_dir)
    print(f"[save] weight file: {wf}  ({os.path.getsize(wf) if wf else 'MISSING'} bytes)")

    nan_found = False
    if wf is not None:
        reloaded = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        for n, p in reloaded.named_parameters():
            if not torch.isfinite(p).all():
                nan_found = True
                print(f"[save] NON-FINITE weights in: {n}")
    print(f"[save] weights reload OK={wf is not None}  NaN/inf present={nan_found}")

    # Final evaluation
    results = trainer.evaluate()
    print("\n=== Final Results (vs the labels it trained on — NOT a before/after) ===")
    for k, v in results.items():
        try:
            print(f"{k}: {v:.4f}")
        except (TypeError, ValueError):
            print(f"{k}: {v}")

    # --- Collapse verdict, printed loud for the orchestrator and the human. -------
    f1 = float(results.get("eval_f1_micro", 0.0))
    ppf = float(results.get("eval_pred_pos_frac", 0.0))
    collapsed = (f1 == 0.0) or (ppf < 1e-6)
    print("\n" + "=" * 70)
    print(f"[VERDICT] eval_f1_micro={f1:.4f}  pred_pos_frac={ppf:.4f}  "
          f"weights_ok={wf is not None}  nan={nan_found}")
    if collapsed:
        print("[VERDICT] *** MODEL COLLAPSED *** — predicts ~nothing-present. NOT trainable "
              "under these settings. (See [run] precision/lr line above.)")
    elif wf is None or nan_found:
        print("[VERDICT] *** WEIGHTS BAD *** — trained but could not save reloadable, "
              "finite weights. Not shippable.")
    else:
        print("[VERDICT] model learned (f1_micro>0, finite weights saved). "
              "Honest before/after is the next step.")
    print("=" * 70, flush=True)
    # Non-zero exit on a dead/broken model so the run script's gate + email report
    # FAILED instead of a green light on a collapsed model.
    if collapsed or wf is None or nan_found:
        raise SystemExit(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    # Precision: default fp32 (both off). The 1 June collapse ran under bf16; fp32 is
    # the first hypothesis the run script tests. Pass --bf16 or --fp16 to compare.
    parser.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="float16 mixed precision (DeBERTa-v3: prone to NaN)")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="LR warmup fraction (stability)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("Pick at most one of --bf16 / --fp16 (default is fp32).")
    main(args)
