"""
Convert binary classification data to DeBERTa multi-label format.

Input format (data/binary/train.jsonl):
    {"text": "concept...", "labels": {"CLAIM": 1, "EVIDENCE": 0, ...}}

Output format (training/data/deberta_train.jsonl):
    {"text": "concept...", "labels": [1, 0, 1, 1, 0]}

Labels array order: [CLAIM, EVIDENCE, SCOPE, ASSUMPTIONS, GAPS]

CRITICAL — GAPS polarity:
    - 1 = gaps are present (negative — concept has reasoning gaps)
    - 0 = no gaps (positive — reasoning is complete)
    This differs from other dimensions where 1 = positive.
"""

import json
from pathlib import Path

DIMENSIONS = ["CLAIM", "EVIDENCE", "SCOPE", "ASSUMPTIONS", "GAPS"]


def convert_to_multilabel(input_path: str, output_path: str) -> dict:
    """
    Convert dict-format labels to array-format labels.

    Returns:
        dict with conversion statistics
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped_missing_labels": 0,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            stats["total"] += 1
            item = json.loads(line)

            # Check all dimensions present
            labels_dict = item.get("labels", {})
            if not all(dim in labels_dict for dim in DIMENSIONS):
                stats["skipped_missing_labels"] += 1
                continue

            # Convert to array format
            labels_array = [labels_dict[dim] for dim in DIMENSIONS]

            output_item = {
                "text": item["text"],
                "labels": labels_array
            }

            f_out.write(json.dumps(output_item) + "\n")
            stats["converted"] += 1

    return stats


def validate_output(output_path: str, sample_size: int = 5) -> None:
    """Print sample outputs for validation."""
    print(f"\n=== Sample outputs from {output_path} ===\n")

    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            item = json.loads(line)
            print(f"Example {i+1}:")
            print(f"  Text: {item['text'][:80]}...")
            print(f"  Labels: {item['labels']}")
            print(f"  Meaning: CLAIM={item['labels'][0]}, EVIDENCE={item['labels'][1]}, "
                  f"SCOPE={item['labels'][2]}, ASSUMPTIONS={item['labels'][3]}, GAPS={item['labels'][4]}")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert to DeBERTa multi-label format")
    parser.add_argument(
        "--input",
        default="data/binary/train.jsonl",
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        default="training/data/deberta_train.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print sample outputs after conversion"
    )
    args = parser.parse_args()

    print(f"Converting {args.input} to {args.output}")
    print(f"Dimension order: {DIMENSIONS}")
    print()

    stats = convert_to_multilabel(args.input, args.output)

    print("=== Conversion Complete ===")
    print(f"Total input lines: {stats['total']}")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Skipped (missing labels): {stats['skipped_missing_labels']}")

    if args.validate:
        validate_output(args.output)
