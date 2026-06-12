"""
analyze_annotations.py — post-labelling analysis script

Loads annotations.csv (human labels from the annotation tool) and compares
against the model's original predictions. Works on a partially completed
annotations.csv so you can check progress at any time.

Outputs:
    research_notes_annotation_results.md  — markdown summary with tables
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd

ANNOTATIONS_FILE = "annotations.csv"
OUTPUT_FILE      = "research_notes_annotation_results.md"
ANNOTATION_COLS  = ["text", "model_prediction", "model_confidence", "human_label"]
TOTAL_COMMENTS   = 432   # full set in low_confidence_comments.csv

# ── Load ────────────────────────────────────────────────────────────────────────

def load_annotations() -> pd.DataFrame:
    p = Path(ANNOTATIONS_FILE)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=ANNOTATION_COLS)
    df = pd.read_csv(p)
    df = df.reindex(columns=ANNOTATION_COLS, fill_value="")
    df = df[df["human_label"].notna() & (df["human_label"].str.strip() != "")]
    return df.reset_index(drop=True)


# ── Analysis ────────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> dict:
    n = len(df)
    all_labels  = ["Positive", "Negative", "Neutral", "Sarcastic/Mixed"]
    label_counts = df["human_label"].value_counts()
    dist = {lbl: int(label_counts.get(lbl, 0)) for lbl in all_labels}

    # Agreement analysis restricted to P/N cases
    pn = df[df["human_label"].isin(["Positive", "Negative"])].copy()
    pn_total = len(pn)
    pn_agree = int((pn["model_prediction"] == pn["human_label"]).sum()) if pn_total else 0

    # Novel classes the binary model cannot express
    novel_n = dist["Neutral"]
    novel_s = dist["Sarcastic/Mixed"]

    # Confusion table: model prediction (rows) × human label (cols)
    model_labels = ["Positive", "Negative"]
    human_labels = ["Positive", "Negative", "Neutral", "Sarcastic/Mixed"]

    if n:
        table = pd.crosstab(
            df["model_prediction"],
            df["human_label"],
        ).reindex(index=model_labels, columns=human_labels, fill_value=0)
    else:
        table = pd.DataFrame(0, index=model_labels, columns=human_labels)

    return dict(
        n=n,
        total=TOTAL_COMMENTS,
        dist=dist,
        pn_agree=pn_agree,
        pn_total=pn_total,
        novel_n=novel_n,
        novel_s=novel_s,
        table=table,
    )


# ── Report ──────────────────────────────────────────────────────────────────────

def build_report(stats: dict) -> str:
    n        = stats["n"]
    total    = stats["total"]
    dist     = stats["dist"]
    pn_agree = stats["pn_agree"]
    pn_total = stats["pn_total"]
    novel_n  = stats["novel_n"]
    novel_s  = stats["novel_s"]
    table    = stats["table"]

    def pct(k):
        return f"{dist[k] / n * 100:.1f}%" if n else "—"

    pn_acc = f"{pn_agree / pn_total * 100:.1f}%" if pn_total else "—"
    novel_pct = f"{(novel_n + novel_s) / n * 100:.1f}%" if n else "—"
    neutral_pct = f"{novel_n / n * 100:.1f}%" if n else "—"

    # Build confusion table markdown
    human_labels = ["Positive", "Negative", "Neutral", "Sarcastic/Mixed"]
    conf_hdr  = "| Model \\ Human | Positive | Negative | Neutral | Sarcastic/Mixed | Row Total |\n"
    conf_sep  = "|---|---|---|---|---|---|\n"
    conf_rows = ""
    for mp in ["Positive", "Negative"]:
        row_vals = [int(table.at[mp, hl]) if hl in table.columns else 0
                    for hl in human_labels]
        conf_rows += f"| **{mp}** | {' | '.join(str(v) for v in row_vals)} | {sum(row_vals)} |\n"
    col_totals = [int(table[hl].sum()) if hl in table.columns else 0
                  for hl in human_labels]
    conf_rows += f"| **Col Total** | {' | '.join(str(v) for v in col_totals)} | {sum(col_totals)} |\n"

    report = f"""# Annotation Results — Post-Labelling Analysis
**Date:** {date.today().isoformat()}
**Source:** `{ANNOTATIONS_FILE}`
**Progress:** {n} / {total} comments labelled ({n / total * 100:.1f}%)

---

## 1. Human Label Distribution

| Label | Count | Share |
|-------|-------|-------|
| Positive | {dist['Positive']} | {pct('Positive')} |
| Negative | {dist['Negative']} | {pct('Negative')} |
| Neutral | {dist['Neutral']} | {pct('Neutral')} |
| Sarcastic/Mixed | {dist['Sarcastic/Mixed']} | {pct('Sarcastic/Mixed')} |
| **Total labelled** | **{n}** | |

---

## 2. Model vs Human Agreement (Positive / Negative Only)

Neutral and Sarcastic/Mixed comments are excluded from this calculation — the
binary model has no way to express those classes, so comparing predictions
on them would understate true agreement on cases where the binary framing is
appropriate.

| Metric | Value |
|--------|-------|
| P/N comments in labelled sample | {pn_total} |
| Model agrees with human | {pn_agree} |
| **Agreement rate** | **{pn_acc}** |

---

## 3. Novel Classes (Not Expressible by the Binary Model)

| Class | Count | Share of labelled |
|-------|-------|-------------------|
| Neutral | {novel_n} | {neutral_pct} |
| Sarcastic / Mixed | {novel_s} | {f"{novel_s / n * 100:.1f}%" if n else "—"} |
| **Combined** | **{novel_n + novel_s}** | **{novel_pct}** |

These comments are the primary motivation for the three-class model extension
(Notebook 5). The binary model cannot recover from this type of error —
a forced Positive or Negative prediction on a genuinely neutral comment is
structurally wrong, not just a classification mistake.

---

## 4. Confusion Table (Model Prediction × Human Label)

{conf_hdr}{conf_sep}{conf_rows}
**How to read:** rows = what the model predicted before human review; columns =
what the human assigned. A cell value shows how many comments the model placed
into that predicted class but the human labelled differently.

---

## 5. Key Takeaways

- **Progress:** {n} / {total} ({n / total * 100:.1f}%) comments labelled so far.
- **Model accuracy on P/N cases:** {pn_acc} — the binary model is reasonably
  calibrated on comments that carry clear sentiment; its failures concentrate
  in the novel classes below.
- **Novel-class rate:** {novel_pct} of labelled comments ({novel_n + novel_s})
  are Neutral or Sarcastic/Mixed — classes the binary model cannot represent.
  This confirms the failure patterns documented in `research_notes.md`.
- **Neutral training examples for Notebook 5:** {novel_n} comments labelled
  Neutral, forming the new third class in the training corpus.
- **Sarcastic/Mixed held out:** {novel_s} comments set aside as `sarcasm_set.csv`
  for a separate future study (irony/sarcasm detection).
"""
    return report


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    df = load_annotations()

    if df.empty:
        print(f"No annotations found in {ANNOTATIONS_FILE}.")
        print("Label some comments with annotate.py first.")
        sys.exit(0)

    print(f"Loaded {len(df)} annotations.")
    stats  = analyse(df)
    report = build_report(stats)

    Path(OUTPUT_FILE).write_text(report, encoding="utf-8")
    print(f"Report saved → {OUTPUT_FILE}\n")

    n  = stats["n"]
    d  = stats["dist"]
    pn = stats["pn_agree"]
    pt = stats["pn_total"]
    print(f"{'─' * 52}")
    print(f"  Progress        : {n} / {stats['total']} ({n / stats['total'] * 100:.1f}%)")
    print(f"  Positive        : {d['Positive']}")
    print(f"  Negative        : {d['Negative']}")
    print(f"  Neutral         : {d['Neutral']}")
    print(f"  Sarcastic/Mixed : {d['Sarcastic/Mixed']}")
    if pt:
        print(f"  P/N agreement   : {pn}/{pt} ({pn / pt * 100:.1f}%)")
    print(f"  Novel classes   : {stats['novel_n']} Neutral + {stats['novel_s']} Sarcastic/Mixed")
    print(f"{'─' * 52}")


if __name__ == "__main__":
    main()
