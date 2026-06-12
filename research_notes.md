# Research Notes — Low-Confidence Analysis
**Date:** 2026-06-12  
**Dataset:** `collected_with_predictions.csv` (4,138 YouTube comments)  
**Model:** XLM-RoBERTa fine-tuned on RUSAD (binary: Positive / Negative)

---

## Summary

432 comments (~10.4%) fell below the 70% confidence threshold and were extracted
to `low_confidence_comments.csv`, sorted lowest-confidence first.

| Confidence band | Count | Share |
|-----------------|-------|-------|
| 50–60% | 174 | 4.2% |
| 60–70% | 258 | 6.2% |
| 70–90% | 866 | 20.9% |
| 90%+ | 2,840 | 68.6% |

The 432 low-confidence comments form a **candidate annotation set** for a future
neutral-class model. Manually labelling these with a three-way schema
(Positive / Neutral / Negative) would provide targeted training signal for the
failure patterns identified below.

---

## Failure Pattern 1 — Missing Neutral Class (Questions and Factual Statements)

The binary model is forced to assign Positive or Negative to comments that carry
no sentiment at all. Questions asking for information and neutral factual
observations are structurally unclassifiable under a two-label scheme, so the
model hedges at ~50%.

**Examples:**

| Confidence | Predicted | Text |
|------------|-----------|------|
| 50.0% | Positive | *"bhai price b bataeya kro sb chezo ke"* |
| 50.6% | Negative | *"Is it safe for the female tourist to visit?"* |
| 50.9% | Positive | *"is drama ka naam kya hai bhai"* |

All three are neutral requests for information. The lack of any sentiment-bearing
vocabulary leaves the model with no signal, producing near-random outputs.

---

## Failure Pattern 2 — Emoji-Marked Sarcasm

Roman Urdu social media frequently uses 😂 and 😅 to invert the literal tone of
a comment. A statement that reads as factual or even positive on the surface
is understood by readers as critical or mocking. The model lacks a mechanism to
detect this pragmatic inversion.

**Examples:**

| Confidence | Predicted | Text |
|------------|-----------|------|
| 50.1% | Positive | *"1971 yaad hai 😂😂😂"* |
| 50.2% | Positive | *"Space technology ka istemal kiya hai 😂😂😂😂😂😂😂😂"* |
| 50.6% | Negative | *"Jab Cricket mein hugte rahenge to us k baad yahi ek kaam rahega in k liye 😂😂"* |

In each case the 😂 signals mockery: the first is a jibe at India-Pakistan
rivalry, the second ridicules a technological claim, the third sarcastically
dismisses the cricket team. The literal text gives mixed or positive signals;
the intended sentiment is negative.

---

## Failure Pattern 3 — Genuinely Mixed Sentiment

Some comments contain both positive and negative content within the same
utterance — a prayer followed by a social critique, a compliment alongside a
complaint, or an expression of sadness framed as appreciative. A binary
classifier must pick one label, producing low confidence rather than
acknowledging the genuine ambivalence.

**Examples:**

| Confidence | Predicted | Text |
|------------|-----------|------|
| 50.1% | Negative | *"Allah sab bache or bachion k naseeb behtreen kare… orat jitni b establish ho jae but ager us ki zindagi me ane wala merd us ki qadar na kare to zindagi bhttttt mushkil he"* |
| 50.6% | Positive | *"Nadia is too critical, but has sooo generous towards stupid drama like Meri Zindagi Hai tu"* |
| 51.0% | Positive | *"pakistan mai jaise hindu behen betiyon ke sath galt hota hai sunkr bohot bura lgta hai apna khayal rkhe / may lord narayan shower blessings upon whole hindu community living in immense hatred in pakistan"* |

The first blends a positive prayer with a bleak assessment of women's social
standing. The second is a mixed drama review. The third opens with a negative
observation and closes with a blessing — opposite sentiments in one comment.

---

## Failure Pattern 4 — Poetic and Indirect Expression

Roman Urdu, particularly in the context of drama and music comments, draws
heavily on Urdu literary conventions: metaphor, wordplay, and indirect emotional
expression. These constructions do not map onto the direct positive/negative
vocabulary seen in training data.

**Examples:**

| Confidence | Predicted | Text |
|------------|-----------|------|
| 51.0% | Positive | *"Zakham puranay ho jae lekin hote tou zakham hain bas hum jina seekh lete hain sath 😌"* |
| 51.8% | Negative | *"Urdu wala safar bhi aur angraizi wala suffer bhi ❤"* |
| 54.8% | Negative | *"Hum Pe Hi Guzre Kyun / Jo Ghum Bate Woh Tumko Na mile / Jiya Hi Dharke Yun Ke Aj Bhi Tera Hi Naam…"* |

The first ("wounds may age but they are still wounds; we just learn to live with
them") expresses melancholic acceptance — arguably negative in affect but
delivered with quiet resignation. The second is a pun: *safar* (journey) vs
*suffer* — ironic and tonally ambiguous. The third is a lyric fragment whose
emotion is conveyed through poetic imagery rather than lexical sentiment markers.

---

## Implications for Future Work

1. **Neutral class extension** — adding a Neutral label and fine-tuning on the
   432-comment annotation set would directly address Pattern 1 and reduce
   forced binary outputs.
2. **Emoji-aware features** — treating emoji sequences as sentiment-modifying
   tokens (or training on sarcasm-labelled data) could address Pattern 2.
3. **Confidence-weighted evaluation** — excluding sub-70% predictions from
   reported accuracy metrics would give a more honest picture of the model's
   reliable operating range (3,706 / 4,138 comments = 89.6% of collected data).
4. **Domain mismatch** — the training corpus (RUSAD reviews) skews toward
   explicit sentiment language; social media comments from cricket, news, and
   music contexts use more indirect registers, suggesting domain-adaptive
   fine-tuning as a priority.
