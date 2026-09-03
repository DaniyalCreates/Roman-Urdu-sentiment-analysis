# Annotation Results — Post-Labelling Analysis
**Date:** 2026-09-03
**Source:** `annotations.csv`
**Progress:** 40 / 432 comments labelled (9.3%)

---

## 1. Human Label Distribution

| Label | Count | Share |
|-------|-------|-------|
| Positive | 9 | 22.5% |
| Negative | 7 | 17.5% |
| Neutral | 14 | 35.0% |
| Sarcastic/Mixed | 10 | 25.0% |
| **Total labelled** | **40** | |

---

## 2. Model vs Human Agreement (Positive / Negative Only)

Neutral and Sarcastic/Mixed comments are excluded from this calculation — the
binary model has no way to express those classes, so comparing predictions
on them would understate true agreement on cases where the binary framing is
appropriate.

| Metric | Value |
|--------|-------|
| P/N comments in labelled sample | 16 |
| Model agrees with human | 9 |
| **Agreement rate** | **56.2%** |

---

## 3. Novel Classes (Not Expressible by the Binary Model)

| Class | Count | Share of labelled |
|-------|-------|-------------------|
| Neutral | 14 | 35.0% |
| Sarcastic / Mixed | 10 | 25.0% |
| **Combined** | **24** | **60.0%** |

These comments are the primary motivation for the three-class model extension
(Notebook 5). The binary model cannot recover from this type of error —
a forced Positive or Negative prediction on a genuinely neutral comment is
structurally wrong, not just a classification mistake.

---

## 4. Confusion Table (Model Prediction × Human Label)

| Model \ Human | Positive | Negative | Neutral | Sarcastic/Mixed | Row Total |
|---|---|---|---|---|---|
| **Positive** | 6 | 4 | 10 | 4 | 24 |
| **Negative** | 3 | 3 | 4 | 6 | 16 |
| **Col Total** | 9 | 7 | 14 | 10 | 40 |

**How to read:** rows = what the model predicted before human review; columns =
what the human assigned. A cell value shows how many comments the model placed
into that predicted class but the human labelled differently.

---

## 5. Key Takeaways

- **Progress:** 40 / 432 (9.3%) comments labelled so far.
- **Model accuracy on P/N cases:** 56.2% — the binary model is reasonably
  calibrated on comments that carry clear sentiment; its failures concentrate
  in the novel classes below.
- **Novel-class rate:** 60.0% of labelled comments (24)
  are Neutral or Sarcastic/Mixed — classes the binary model cannot represent.
  This confirms the failure patterns documented in `research_notes.md`.
- **Neutral training examples for Notebook 5:** 14 comments labelled
  Neutral, forming the new third class in the training corpus.
- **Sarcastic/Mixed held out:** 10 comments set aside as `sarcasm_set.csv`
  for a separate future study (irony/sarcasm detection).
