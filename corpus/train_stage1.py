import argparse
import json
import torch
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

    return accuracies

def main(args):
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
        fp16=True,  # Half precision - no accuracy loss, halves memory
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_mean",
        logging_steps=50,
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

    # Final evaluation
    results = trainer.evaluate()
    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    main(args)
