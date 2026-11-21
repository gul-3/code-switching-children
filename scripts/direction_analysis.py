"""Descriptive analyses by switch direction (session language), main sample only.

Backs the numbers in results/RESULTS.md section 6 and docs/FINDINGS.md. Nothing here trains a model; it
re-reads the dataset and the out-of-fold predictions written by run_models.py.

  1. Label counts by corpus x session language (how often children switch, in which direction).
  2. Out-of-fold scores of M1 (other) and M3 (mixed) per session language.
  3. Language of the preceding adult utterance for child 'other' utterances.
  4. Determiner pattern (same definition and lexicon as error_analysis.py) by corpus x session language.
  5. Lexical diversity of the child's 'other' utterances per direction:
     number of word types, tokens, and the share of 'other' utterances that contain at
     least one of the 10 most frequent word types of that direction.

No significance tests are run; all numbers are counts or proportions.
Usage: .venv/bin/python scripts/direction_analysis.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "direction_analysis"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(ROOT / "data" / "all_corpora.csv", keep_default_na=False, dtype=str)
d = d[d["main_sample"] == "1"]
o = pd.read_csv(ROOT / "results" / "oof_predictions.csv", keep_default_na=False)


def save(t, name):
    t.to_csv(OUT / f"{name}.csv")
    print(f"\n== {name}\n{t.to_string()}")


# 1. counts
save(pd.crosstab([d["corpus"], d["session_language"]], d["label3"], margins=True), "labels_by_corpus_session")
share = pd.crosstab([d["corpus"], d["session_language"]], d["label3"], normalize="index").mul(100).round(1)
save(share, "labels_by_corpus_session_pct")

# 2. per-direction scores (out-of-fold)
rows = []
for corpus in ["all"] + sorted(o["corpus"].unique()):
    for s in ["eng", "spa"]:
        x = o[(o["session_language"] == s) & ((o["corpus"] == corpus) | (corpus == "all"))]
        if (x["label3"] == "other").sum() == 0:
            continue
        rows.append({
            "corpus": corpus, "session_language": s, "n": len(x),
            "n_other": int((x["label3"] == "other").sum()), "n_mixed": int((x["label3"] == "mixed").sum()),
            "M1_f1_other": round(f1_score(x["label3"] == "other", x["pred_M1_lexical"] == "other"), 3),
            "M3_f1_other": round(f1_score(x["label3"] == "other", x["pred_M3_lex_plus_gram"] == "other"), 3),
            "M1_f1_mixed": round(f1_score(x["label3"] == "mixed", x["pred_M1_lexical"] == "mixed"), 3),
            "M3_f1_mixed": round(f1_score(x["label3"] == "mixed", x["pred_M3_lex_plus_gram"] == "mixed"), 3),
        })
save(pd.DataFrame(rows).set_index(["corpus", "session_language"]), "scores_by_direction")

# 3. preceding adult utterance
save(pd.crosstab([d["corpus"], d["label3"]], d["prev_adult_language"], margins=True), "prev_adult_by_label")
d["prev_is_context"] = d["prev_adult_language"] == d["session_language"]
save(pd.crosstab(d["label3"], d["prev_is_context"], normalize="index").mul(100).round(1),
     "prev_adult_is_context_language_pct")
oth = d[d["label3"] == "other"]
oth_lang = oth["session_language"].map({"eng": "spa", "spa": "eng"})
save(pd.crosstab(oth["corpus"], oth["prev_adult_language"] == oth_lang, margins=True)
     .rename(columns={True: "adult_in_other_language", False: "adult_not_in_other_language"}),
     "other_after_adult_in_other_language")

# 4. determiner pattern by session language (reuses error_analysis lexicon definition)
import error_analysis as ea  # noqa: E402  (runs error_analysis once; it only writes its own tables)

mixed = o[o["label3"] == "mixed"].copy()
mixed["det_pattern"] = mixed["clean"].map(ea.det_pattern)
save(pd.crosstab([mixed["corpus"], mixed["session_language"]], mixed["det_pattern"], margins=True),
     "determiner_pattern_by_corpus_session")

# 5. lexical diversity of 'other' utterances
rows = []
for s in ["eng", "spa"]:
    x = d[(d["label3"] == "other") & (d["session_language"] == s)]
    tokens = [w for t in x["clean"].str.lower() for w in t.split()]
    c = Counter(tokens)
    top10 = [w for w, _ in c.most_common(10)]
    covered = x["clean"].str.lower().map(lambda t: any(w in top10 for w in t.split())).mean()
    rows.append({"session_language": s, "utterance_language": "spa" if s == "eng" else "eng",
                 "n_utterances": len(x), "tokens": len(tokens), "types": len(c),
                 "types_per_100_tokens": round(100 * len(c) / len(tokens), 1),
                 "share_utts_with_top10_word": round(covered, 3), "top10": " ".join(top10)})
save(pd.DataFrame(rows).set_index("session_language"), "other_lexical_diversity")
print("\nwrote", OUT)
