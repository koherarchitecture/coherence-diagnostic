# Annotation Schema — Coherence Diagnostic

**Tool:** Coherence Diagnostic
**Stage 1 model:** DeBERTa-v3-base, fine-tuned for multi-label classification
**Purpose of this document:** Specify how design concept statements were annotated for Stage 1 training. Accompanies `aggregate-performance.md`.

---

## The five dimensions

Each concept statement was annotated on five independent binary labels.

| Dimension | Annotator question | Polarity |
|-----------|--------------------|----------|
| CLAIM | Is the core claim clearly stated? | Standard (1 = present) |
| EVIDENCE | Is there adequate supporting evidence? | Standard (1 = present) |
| SCOPE | Is the scope appropriately bounded? | Standard (1 = present) |
| ASSUMPTIONS | Are key assumptions acknowledged? | Standard (1 = present) |
| GAPS | Are there critical reasoning gaps? | Inverted (1 = gaps present = problem) |

The GAPS dimension is the only one with inverted polarity. For the other four, a positive label marks the dimension as present / strong. For GAPS, a positive label marks a problem.

---

## Three-state annotation, binary training labels

Annotators worked in a three-state rubric; labels were then collapsed to binary for training.

| Annotator state | Meaning | Binary label (for four standard dimensions) |
|-----------------|---------|----------------------------------------------|
| SOLID | Dimension is clearly present | 1 |
| WORTH_EXAMINING | Borderline; present but unclear or thin | Reviewed case-by-case; typically 1 |
| ATTENTION_NEEDED | Dimension absent or unrecognisable | 0 |

For GAPS (inverted polarity), the mapping is: SOLID (no gaps) = 0, ATTENTION_NEEDED (gaps present) = 1.

The three-state rubric exists because borderline cases carry more training signal than either extreme. WORTH_EXAMINING annotations were reviewed individually before being committed to binary labels.

---

## Dimension definitions and examples

### CLAIM — Is the core claim clearly stated?

The question the annotator asks themselves: *"Can I picture what this design will actually be?"*

| State | What it looks like |
|-------|--------------------|
| SOLID | The specific design is described with enough detail that it could be sketched. Example: *"I am designing a wayfinding system for elderly hospital visitors that uses colour-coded floor markers instead of digital screens."* |
| WORTH_EXAMINING | The domain is named but the design itself is not. Example: *"I want to help people navigate hospitals better using design."* |
| ATTENTION_NEEDED | The statement describes a problem or aspiration but never says what the design *is*. |

### EVIDENCE — Is there adequate supporting evidence?

The question: *"Does the author show how they know the claim is true?"*

| State | What it looks like |
|-------|--------------------|
| SOLID | Concrete observation, data, interviews, or research is cited and tied to the claim. |
| WORTH_EXAMINING | Evidence is referenced but thin, vague, or not clearly connected to the claim. |
| ATTENTION_NEEDED | The claim is asserted without support; assumptions stand in for evidence. |

### SCOPE — Is the scope appropriately bounded?

The question: *"Is it clear who this is for, where it applies, and where it doesn't?"*

| State | What it looks like |
|-------|--------------------|
| SOLID | User group, context, and boundaries of the design are explicit. |
| WORTH_EXAMINING | Scope is mentioned but vague; boundaries are not clearly drawn. |
| ATTENTION_NEEDED | The scope is unbounded or unstated; the design could apply to anyone, anywhere. |

### ASSUMPTIONS — Are key assumptions acknowledged?

The question: *"Does the author name what must be true for the design to work?"*

| State | What it looks like |
|-------|--------------------|
| SOLID | Key assumptions are explicitly acknowledged, even if not yet tested. |
| WORTH_EXAMINING | Some assumptions are gestured at but not clearly stated. |
| ATTENTION_NEEDED | The statement proceeds as if its assumptions are self-evident. |

### GAPS — Are there critical reasoning gaps?

The question: *"Do the steps from problem to solution hold together?"*

| State | What it looks like |
|-------|--------------------|
| SOLID (no gaps) | The reasoning chain from problem to proposed design is connected and inspectable. |
| WORTH_EXAMINING | Some connections are unclear; the logic is partially stated. |
| ATTENTION_NEEDED (gaps present) | There are logical jumps; the proposed design does not follow from the stated problem or evidence. |

---

## Confidence thresholds at inference time

At inference time, the five DeBERTa classifiers output confidence scores in [0.0, 1.0]. These are converted to three severity levels by the Stage 2 rules layer (`src/stage2_rules.py`):

**Standard polarity (CLAIM, EVIDENCE, SCOPE, ASSUMPTIONS):**

| Confidence | Severity | Display |
|------------|----------|---------|
| > 0.8 | SOLID | ● Present |
| 0.5 – 0.8 | WORTH_EXAMINING | ◐ Unclear |
| < 0.5 | ATTENTION_NEEDED | ○ Missing |

**Inverted polarity (GAPS):**

| Confidence | Severity | Display |
|------------|----------|---------|
| < 0.2 | SOLID | ● Connected |
| 0.2 – 0.5 | WORTH_EXAMINING | ◐ Unclear |
| > 0.5 | ATTENTION_NEEDED | ○ Gaps present |

Thresholds (0.8, 0.5, 0.2) are design decisions encoded in code, not empirically derived. They are fully inspectable and adjustable — change the threshold, change the behaviour.

---

## What this schema does not cover

- **Design quality.** The five dimensions capture coherence properties — internal structure of a concept statement. They do not assess whether the design is good, ethical, creative, or worth pursuing. Those are human judgements the tool does not claim.
- **Tacit knowledge.** Aspects of teaching presence (timing, tone, reading the room) are not captured. The schema captures the articulable part of structural reading only.
- **Aesthetic or cultural evaluation.** Not in scope.

---

## Corpus

Training corpus: ~4,186 annotated design concept statements from Anant National University. Due to student-privacy commitments, the corpus itself is not publicly released. This schema document and the accompanying `aggregate-performance.md` are released so that the annotation process and its results are inspectable.

---

*Last updated: 17 April 2026*
