"""Error analysis of the 'mixed' class for the best model (M3, lexical + grammatical).

Uses the out-of-fold predictions in results/oof_predictions.csv, i.e. every
prediction was made by a model that never saw that child.

Word-language lexicon (descriptive only, not used by any model):
  The corpora mark language per utterance, not per word. To describe which words in
  an utterance belong to which language, every word is assigned a language from
  how often it occurs in monolingual utterances (label 'same' or 'other', whose
  language is known): Spanish if >85% of its occurrences are in Spanish utterances,
  English if <15%, 'both' in between, 'unk' if seen in fewer than 3 utterances.
  This is a heuristic; it is used only to group errors, never for evaluation.

Writes results/error_analysis/*.csv and prints the tables used in ERROR_ANALYSIS.md.
Usage: .venv/bin/python scripts/error_analysis.py
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)
P = "pred_M3_lex_plus_gram"

o = pd.read_csv(ROOT / "results" / "oof_predictions.csv", keep_default_na=False)
o["other_lang"] = o["session_language"].map({"eng": "spa", "spa": "eng"})

# ---------------------------------------------------------------- word lexicon
counts = defaultdict(lambda: [0, 0])          # [n English utts, n Spanish utts]
for text, ctx, lab, oth in zip(o["clean"], o["session_language"], o["label3"], o["other_lang"]):
    if lab == "mixed":
        continue
    lang = ctx if lab == "same" else oth
    for w in set(text.lower().split()):
        counts[w][lang == "spa"] += 1


def word_lang(w):
    eng, spa = counts.get(w, (0, 0))
    if eng + spa < 3:
        return "unk"
    r = spa / (eng + spa)
    return "spa" if r > 0.85 else "eng" if r < 0.15 else "both"


def lang_counts(text, ctx):
    langs = [word_lang(w) for w in text.lower().split()]
    n_ctx = sum(x == ctx for x in langs)
    n_oth = sum(x in ("eng", "spa") and x != ctx for x in langs)
    return n_ctx, n_oth


def table(rows, cols, name, normalize=False):
    t = pd.crosstab(rows, cols, margins=not normalize, normalize="index" if normalize else False)
    t = t.round(2) if normalize else t
    t.to_csv(OUT / f"{name}.csv")
    print(f"\n== {name}\n{t.to_string()}")
    return t


mixed = o[o["label3"] == "mixed"].copy()
tp = ((o[P] == "mixed") & (o["label3"] == "mixed")).sum()
print(f"mixed: n={len(mixed)}  precision={tp / (o[P] == 'mixed').sum():.3f}  "
      f"recall={tp / len(mixed):.3f}")

# 1. where do missed mixed utterances go, by corpus and length
table(mixed["corpus"], mixed[P], "mixed_by_corpus")
table(mixed["n_tokens"].clip(upper=6), mixed[P], "mixed_by_length")

# 2. composition of the mixed utterance: how many words of each language
nc = mixed.apply(lambda r: lang_counts(r["clean"], r["session_language"]), axis=1)
mixed["n_ctx"], mixed["n_oth"] = zip(*nc)
mixed["composition"] = np.select(
    [mixed["n_oth"] == 0, mixed["n_ctx"] == 0, mixed["n_oth"] == 1, mixed["n_ctx"] == 1],
    ["A no clear other-language word", "B no clear context-language word",
     "C one other-language word", "D one context-language word"],
    "E balanced (2+ of each)")
table(mixed["composition"], mixed[P], "mixed_by_composition")
table(mixed["composition"], mixed[P], "mixed_by_composition_share", normalize=True)

# 3. functional-lexical mixing: determiner of one language + word of the other
SPA_DET = {"un", "una", "el", "la", "los", "las", "mi", "este", "esta", "otro", "otra"}
ENG_DET = {"a", "the", "this", "my", "that", "another"}


def det_pattern(text):
    w = text.lower().split()
    for a, b in zip(w, w[1:]):
        if a in SPA_DET and word_lang(b) == "eng":
            return "Spanish determiner + English word"
        if a in ENG_DET and word_lang(b) == "spa":
            return "English determiner + Spanish word"
    return "other pattern"


mixed["det_pattern"] = mixed["clean"].map(det_pattern)
table(mixed["det_pattern"], mixed[P], "mixed_by_determiner_pattern")
table(mixed["det_pattern"], mixed[P], "mixed_by_determiner_pattern_share", normalize=True)
table(mixed["det_pattern"], mixed["corpus"], "determiner_pattern_by_corpus")

# 4. false positives: 'same' utterances predicted as mixed
fp = o[(o[P] == "mixed") & (o["label3"] == "same")].copy()
nc = fp.apply(lambda r: lang_counts(r["clean"], r["session_language"]), axis=1)
fp["n_ctx"], fp["n_oth"] = zip(*nc)
fp["kind"] = np.select(
    [(fp["n_oth"] > 0) & (fp["n_ctx"] > 0), (fp["n_oth"] > 0) & (fp["n_ctx"] == 0)],
    ["contains words of both languages", "only other-language words"],
    "no other-language word")
table(fp["corpus"], fp["kind"], "false_mixed_by_corpus")

# 5. examples for reading
cols = ["corpus", "session_language", "clean", "label3", P]
mixed.sample(frac=1, random_state=1).groupby(["composition", P]).head(8)[cols + ["composition", "det_pattern"]] \
    .sort_values(["composition", P]).to_csv(OUT / "examples_mixed.csv", index=False)
fp.sample(frac=1, random_state=1).groupby(["corpus", "kind"]).head(8)[cols + ["kind"]] \
    .sort_values(["corpus", "kind"]).to_csv(OUT / "examples_false_mixed.csv", index=False)
print("\nwrote", OUT)
