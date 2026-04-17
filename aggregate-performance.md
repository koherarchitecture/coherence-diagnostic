# Aggregate Performance Metrics — Coherence Diagnostic

**Tool:** Coherence Diagnostic
**Stage 1 model:** DeBERTa-v3-base, fine-tuned for multi-label classification
**Purpose of this document:** Report per-classifier and aggregate validation accuracy. Accompanies `annotation-schema.md`.

---

## Model

| Property | Value |
|----------|-------|
| Base model | microsoft/deberta-v3-base |
| Task | Multi-label binary classification (5 labels) |
| Training date | 11 January 2026 |
| Training corpus size | ~4,186 annotated design concept statements |
| Annotation schema | See `annotation-schema.md` |
| Model size | ~738 MB (model.safetensors) |

---

## Per-classifier validation accuracy

The five dimensions are jointly trained as a single multi-label DeBERTa head (`num_labels=5`, `problem_type="multi_label_classification"`). Per-dimension accuracies below are computed independently on the held-out evaluation split.

**Validation methodology.** The 4,186-statement training corpus was split internally 90/10 at random (`torch.utils.data.random_split`), giving ~3,768 training and ~419 held-out evaluation statements. The model was trained for 5 epochs; the best checkpoint (selected by mean per-dimension accuracy) was at **epoch 4**. The numbers below come from that checkpoint.

| Classifier | Accuracy |
|------------|----------|
| CLAIM | 97.85% |
| EVIDENCE | 98.81% |
| SCOPE | 99.28% |
| ASSUMPTIONS | 97.37% |
| GAPS | 98.57% |

Range: **97.37%–99.28%**.

---

## Aggregate

| Aggregation | Value |
|-------------|-------|
| Arithmetic mean across the five classifiers | **98.38%** |
| F1 micro (multi-label, all dimensions) | **97.91%** |

The 98.38% figure is the arithmetic mean of the five per-classifier accuracies. Calculation: (97.85 + 98.81 + 99.28 + 97.37 + 98.57) / 5 = 98.376%, rounded to 98.38%. This is not a sample-weighted aggregate; it is a simple mean across the five classifier accuracies reported above.

The F1 micro of 97.91% is a stricter aggregate for multi-label classification — it pools predictions across all dimensions before computing precision and recall, so it accounts for the joint label distribution rather than treating dimensions independently.

---

## What these numbers mean and what they do not

### What they mean

- The DeBERTa classifier, fine-tuned on the ~4,186-statement annotated corpus, correctly labels held-out concept statements on each of the five dimensions at the rates above.
- The range 97.37%–99.28% is the performance envelope across dimensions. ASSUMPTIONS (97.37%) is the hardest dimension to classify; SCOPE (99.28%) is the easiest.
- Each classifier is a Stage 1 output. The Stage 2 rules layer (`src/stage2_rules.py`) converts these confidences into severity levels (SOLID / WORTH_EXAMINING / ATTENTION_NEEDED) using design-chosen thresholds. Rule-layer behaviour is not captured in a single accuracy figure and is not intended to be — rules encode design decisions, not empirical claims.

### What they do not mean

- **They do not mean the Coherence Diagnostic has 98.38% accuracy.** That figure belongs to one Stage 1 of one tool. Treating it as a summary property of the tool, let alone of Koher, would over-claim.
- **They do not establish pedagogical efficacy.** Classifier accuracy measures whether the model agrees with the annotator on a held-out split. Whether students learn better from the tool's output is a separate empirical question the tool has not yet answered.
- **They do not generalise to other institutions.** The corpus comes from one institution (Anant National University). Cross-institutional validation has not yet been run.
- **They do not cover the rules layer.** The rules in `src/stage2_rules.py` are not subject to validation in the classifier sense — they encode pedagogical judgement in code. Whether a threshold of 0.8 for "solid" is correct is a judgement call, not a testable hypothesis.

---

## Reproducibility

- The trained model weights are in `models/deberta-coherence/` in this repository.
- The Stage 2 rules (thresholds, severity mapping, cross-dimension logic) are in `src/stage2_rules.py`.
- The annotation schema used to produce the training labels is in `annotation-schema.md`.
- Training hyperparameters: base model `microsoft/deberta-v3-base`, learning rate 2e-5, batch size 8, 5 epochs, fp16, `metric_for_best_model="accuracy_mean"`, `load_best_model_at_end=True`.
- The corpus itself (~4,186 annotated concept statements) is not publicly released due to student-privacy commitments from Anant National University. The schema and aggregate metrics are released so that the process and its results are inspectable even without access to the underlying data.

**A note on a separately-prepared validation file.** An earlier data-preparation pass produced a 723-statement `val.jsonl` alongside `train.jsonl`. The actual DeBERTa training run did not use that file — it took the 4,186-statement training file and made its own random 90/10 split. The 723-statement file is preserved in the codebase but is not the basis for any number reported here.

---

*Last updated: 17 April 2026*
