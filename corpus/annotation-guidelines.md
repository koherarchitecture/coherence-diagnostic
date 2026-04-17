# Annotation Guidelines for Design Coherence Dimensions

*Version 1.0 — 27 December 2025*

---

## Overview

These guidelines clarify the criteria for labelling the 5 binary dimensions used in the Koher design coherence pipeline. Consistent annotation is critical for training accuracy.

---

## ASSUMPTIONS Dimension

### Question
> Are key assumptions acknowledged?

### Criteria for ASSUMPTIONS = yes (1)

The concept must **explicitly acknowledge** assumptions using language such as:

- "We assume..."
- "This assumes..."
- "The intervention assumes..."
- "Our assumption is..."
- "Assuming that..."

### Criteria for ASSUMPTIONS = no (0)

- No explicit acknowledgement of assumptions
- Implicit causal assumptions stated as facts do **NOT** qualify
- Unstated assumptions embedded in the logic do **NOT** qualify

### Examples

| Text | Label | Rationale |
|------|-------|-----------|
| "We assume farmers will adopt intensive management despite labour requirements." | **yes** | Explicit "We assume..." |
| "Farmers will adopt intensive management despite labour requirements." | **no** | Stated as fact, not acknowledged as assumption |
| "The intervention assumes blood banks provide compatible platelets promptly." | **yes** | Explicit "assumes" |
| "Blood banks provide compatible platelets promptly." | **no** | Stated as fact |
| "Customers will obviously prefer handmade products over factory-made alternatives." | **yes** | "obviously" signals awareness this is an assumption |
| "Crop insurance claims require damage documentation farmers cannot provide." | **no** | Stated as fact, even though it's actually an assumption |

### Common Mistakes

1. **Implicit ≠ Acknowledged:** A concept may contain many implicit assumptions, but ASSUMPTIONS=yes requires explicit acknowledgement.

2. **Problem statements ≠ Assumptions:** Stating "X is a problem" is not acknowledging an assumption unless phrased as "We assume X is a problem."

3. **Causal claims ≠ Acknowledged assumptions:** "Training leads to improved outcomes" is a causal claim, not an acknowledged assumption. "We assume training leads to improved outcomes" would be.

---

## CLAIM Dimension

### Question
> Is the core claim clearly stated?

### Criteria for CLAIM = yes (1)

- A specific, testable statement about what the intervention does or achieves
- Concrete language about outcomes, effects, or deliverables
- Quantified or quantifiable assertions

### Criteria for CLAIM = no (0)

- Vague, aspirational language ("hopes to", "aims to", "could")
- Problem statements without intervention claims
- Marketing language without specifics ("works", "helps", "supports")
- Future intentions without concrete claims

### Examples

| Text | Label | Rationale |
|------|-------|-----------|
| "Our programme trained 500 farmers in sustainable practices." | **yes** | Specific, quantified |
| "Our programme helps farmers." | **no** | Vague |
| "The waste management solution works for various city sizes." | **no** | Marketing language, no specifics |
| "Yield monitoring confirmed 35% increase for adopting farmers." | **yes** | Quantified outcome |
| "We hope to create meaningful impact." | **no** | Aspirational |

---

## EVIDENCE Dimension

### Question
> Is there adequate supporting evidence?

### Criteria for EVIDENCE = yes (1)

- Quantified data or metrics
- Documented outcomes or results
- Procedural evidence (methodology described)
- Comparative data

### Criteria for EVIDENCE = no (0)

- No supporting data
- Vague references ("research indicates", "studies show")
- Anecdotes without systematic documentation
- Future evidence ("will be measured")

---

## SCOPE Dimension

### Question
> Is the scope appropriately bounded?

### Criteria for SCOPE = yes (1)

- Geographic boundaries stated
- Population/beneficiary count specified
- Clear inclusion/exclusion criteria
- Temporal bounds if relevant

### Criteria for SCOPE = no (0)

- Unbounded claims ("across India", "everywhere")
- Missing geographic specificity
- No population count or estimate
- Vague descriptors ("various", "many", "communities")

### Edge Cases

| Text | Label | Rationale |
|------|-------|-----------|
| "flood-prone Bihar districts" | **borderline** | Regional but no specific districts |
| "150 operators in West Bengal's North 24 Parganas district" | **yes** | Count + specific district |
| "rural households" | **no** | No geography or count |

---

## GAPS Dimension

### Question
> Are there critical reasoning gaps?

### Criteria for GAPS = yes (1)

- Explicit acknowledgement of limitations in reasoning
- Stated unknowns or uncertainties
- Recognised logical jumps that need investigation
- Self-aware about what the intervention does NOT address

### Criteria for GAPS = no (0)

- No acknowledged reasoning gaps
- Reasoning chain appears connected (even if flawed)
- Simple statements without complex reasoning to evaluate

### Important Distinction

**Scope exclusions ≠ Reasoning gaps**

| Text | Label | Rationale |
|------|-------|-----------|
| "The service excludes agency-mediated situations." | **no** | Scope exclusion, not reasoning gap |
| "How improved awareness leads to sustained change requires investigation." | **yes** | Acknowledged reasoning gap |
| "Long-term outcomes require tracking beyond this study." | **yes** | Acknowledged limitation |

---

## Audit Trail

When revising labels, add an `audit_note` field to the JSON:

```json
{
  "text": "...",
  "labels": {...},
  "original_tags": [...],
  "audit_note": "ASSUMPTIONS revised 1→0 on 27-Dec-2025: No explicit acknowledgement"
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 27 Dec 2025 | Initial guidelines based on integration test audit |
