"""Train and evaluate the baselines and models of METHODS.md section 11.

Data   : data/all_corpora_tagged.csv, main sample only (ages <= 3;6)
Folds  : data/folds.csv (5 folds, grouped by child)
Target : label3 (same / other / mixed)

Every model is a Logistic Regression with class_weight="balanced" (classes are
weighted inversely to their frequency, so the rare 'mixed' class is not ignored).
Hyperparameters are fixed (C=1.0), not tuned, so no test data influences them.
All feature extractors (TF-IDF vocabularies, encoders) are fitted on the training
folds only, inside the pipeline.

Outputs (results/):
  cv_folds.csv           per model x fold: macro-F1, per-class F1, accuracy
  cv_summary.csv         mean and SD over folds
  oof_predictions.csv    out-of-fold prediction of every model for every utterance
  by_corpus.csv          macro-F1 and per-class F1 per corpus (out-of-fold)
  by_age.csv             same, per age (months)
  by_length.csv          same, per utterance length
  confusion_<model>.csv  summed confusion matrix over the 5 test folds
  loco_ferfulice.csv     leave-one-corpus-out: train without FerFuLice, test on it
  top_features.txt       largest coefficients per class (interpretation)

Usage:  .venv/bin/python scripts/run_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)
CLASSES = ["same", "other", "mixed"]
CONTEXT = ["session_language", "prev_adult_language"]

# ---------------------------------------------------------------- data

df = pd.read_csv(DATA / "all_corpora_tagged.csv", dtype=str, keep_default_na=False)
df = df[df["main_sample"] == "1"].reset_index(drop=True)
folds = pd.read_csv(DATA / "folds.csv", dtype=str)
df = df.merge(folds[["child_id", "fold"]], on="child_id", how="left", validate="m:1")
assert df["fold"].notna().all()
df["fold"] = df["fold"].astype(int)
df["n_tokens"] = df["n_tokens"].astype(int)
df["length_bin"] = df["n_tokens"].clip(upper=6).astype(str)       # 1..5, 6 = 6+


# ---------------------------------------------------------------- tag "documents"

def upos(tags):
    return [t.split("|")[1] for t in tags.split()]


def grammar_doc(row):
    """Model 2a: per tagger, UPOS, UPOS+each morphological feature, UPOS bigrams."""
    out = []
    for prefix, col in (("en", "tags_en"), ("es", "tags_es")):
        toks = row[col].split()
        pos = upos(row[col])
        for t, p in zip(toks, pos):
            out.append(f"{prefix}:{p}")
            feats = t.split("|")[2:]
            out += [f"{prefix}:{p}:{f}" for f in feats if f != "_"]
        seq = ["<s>"] + pos + ["</s>"]
        out += [f"{prefix}:{a}_{b}" for a, b in zip(seq, seq[1:])]
    return " ".join(out)


def pos_seq_doc(row):
    """Model 2b: UPOS uni-, bi- and trigrams per tagger. No words, no features."""
    out = []
    for prefix, col in (("en", "tags_en"), ("es", "tags_es")):
        seq = ["<s>"] + upos(row[col]) + ["</s>"]
        out += [f"{prefix}:{p}" for p in seq[1:-1]]
        out += [f"{prefix}:{a}_{b}" for a, b in zip(seq, seq[1:])]
        out += [f"{prefix}:{a}_{b}_{c}" for a, b, c in zip(seq, seq[1:], seq[2:])]
    return " ".join(out)


df["grammar_doc"] = df.apply(grammar_doc, axis=1)
df["pos_seq_doc"] = df.apply(pos_seq_doc, axis=1)


def word_doc(text):
    """Model 1: lowercased word unigrams and bigrams."""
    w = text.lower().split()
    return " ".join(w + [f"{a}_{b}" for a, b in zip(w, w[1:])])


df["word_doc"] = df["clean"].map(word_doc)

# Context conjunction.  Whether an utterance is a switch depends on the
# COMBINATION of its words and the context: "mira" is 'same' in a Spanish session
# but 'other' in an English one. A linear model with separate word and context
# features cannot represent this interaction (a first run without it gave F1 0.28
# for 'other'). So every linguistic feature is given twice: once as is (shared
# across contexts) and once prefixed with the session language (eng§mira,
# spa§mira), letting each feature have a context-specific weight.
for col in ["word_doc", "grammar_doc", "pos_seq_doc"]:
    df[col + "_ctx"] = [
        doc + " " + " ".join(f"{lang}§{tok}" for tok in doc.split())
        for doc, lang in zip(df[col], df["session_language"])
    ]

# ---------------------------------------------------------------- models

def tfidf(col, **kw):
    return (col, TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                                 sublinear_tf=True, **kw), col)


def onehot(cols):
    return ("ctx_" + "_".join(cols), OneHotEncoder(handle_unknown="ignore"), cols)


def logreg(*blocks):
    return Pipeline([
        ("features", ColumnTransformer(list(blocks))),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)),
    ])


WORDS = tfidf("word_doc_ctx", min_df=2)
GRAMMAR = tfidf("grammar_doc_ctx", min_df=2)
POSSEQ = tfidf("pos_seq_doc_ctx", min_df=2)

MODELS = {
    "B0_majority": DummyClassifier(strategy="most_frequent"),
    "B1_length": logreg(onehot(["length_bin"])),
    "B2_context": logreg(onehot(CONTEXT)),
    "M1_lexical": logreg(WORDS, onehot(CONTEXT)),
    "M2a_grammatical": logreg(GRAMMAR, onehot(CONTEXT)),
    "M2b_pos_sequence": logreg(POSSEQ, onehot(CONTEXT)),
    "M3_lex_plus_gram": logreg(WORDS, GRAMMAR, onehot(CONTEXT)),
}
INPUT_COLS = ["word_doc_ctx", "grammar_doc_ctx", "pos_seq_doc_ctx", "length_bin"] + CONTEXT


def scores(y, p):
    f1 = f1_score(y, p, labels=CLASSES, average=None, zero_division=0)
    return {"macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p),
            **{f"f1_{c}": v for c, v in zip(CLASSES, f1)}, "n": len(y)}


# ---------------------------------------------------------------- 5-fold CV

rows, oof = [], pd.DataFrame(index=df.index)
for name, model in MODELS.items():
    pred = pd.Series(index=df.index, dtype=object)
    for k in range(5):
        tr, te = df["fold"] != k, df["fold"] == k
        model.fit(df.loc[tr, INPUT_COLS], df.loc[tr, "label3"])
        pred[te] = model.predict(df.loc[te, INPUT_COLS])
        rows.append({"model": name, "fold": k, **scores(df.loc[te, "label3"], pred[te])})
    oof[name] = pred
    cm = confusion_matrix(df["label3"], pred, labels=CLASSES)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(OUT / f"confusion_{name}.csv")
    print(f"{name:18s} macro-F1 per fold:",
          [round(r["macro_f1"], 3) for r in rows if r["model"] == name])

cv = pd.DataFrame(rows)
cv.to_csv(OUT / "cv_folds.csv", index=False)
metrics = ["macro_f1", "f1_same", "f1_other", "f1_mixed", "accuracy"]
summary = cv.groupby("model", sort=False)[metrics].agg(["mean", "std"]).round(3)
summary.to_csv(OUT / "cv_summary.csv")
print("\n", summary)

keep = ["corpus", "child_id", "session_language", "prev_adult_language",
        "analysis_age_months", "n_tokens", "clean", "label3", "fold"]
pd.concat([df[keep], oof.add_prefix("pred_")], axis=1).to_csv(OUT / "oof_predictions.csv", index=False)

# ---------------------------------------------------------------- breakdowns (out-of-fold)

def breakdown(by):
    out = []
    for name in MODELS:
        for val, g in df.groupby(by):
            out.append({"model": name, by: val, **scores(g["label3"], oof.loc[g.index, name])})
    return pd.DataFrame(out).round(3)


breakdown("corpus").to_csv(OUT / "by_corpus.csv", index=False)
breakdown("analysis_age_months").to_csv(OUT / "by_age.csv", index=False)
breakdown("length_bin").to_csv(OUT / "by_length.csv", index=False)

# ---------------------------------------------------------------- leave-one-corpus-out

loco = []
tr, te = df["corpus"] != "ferfulice", df["corpus"] == "ferfulice"
for name, model in MODELS.items():
    model.fit(df.loc[tr, INPUT_COLS], df.loc[tr, "label3"])
    loco.append({"model": name, "train": "hoff+deuchar+perez", "test": "ferfulice<=3;6",
                 **scores(df.loc[te, "label3"], model.predict(df.loc[te, INPUT_COLS]))})
loco = pd.DataFrame(loco).round(3)
loco.to_csv(OUT / "loco_ferfulice.csv", index=False)
print("\nLeave-one-corpus-out (test = FerFuLice <= 3;6)\n", loco[["model"] + metrics])

# ---------------------------------------------------------------- interpretation

with open(OUT / "top_features.txt", "w", encoding="utf-8") as fh:
    for name in ["M1_lexical", "M2a_grammatical", "M2b_pos_sequence"]:
        model = MODELS[name].fit(df[INPUT_COLS], df["label3"])     # refit on all data
        feats = model.named_steps["features"].get_feature_names_out()
        clf = model.named_steps["clf"]
        fh.write(f"===== {name} (fitted on the whole main sample)\n")
        for ci, cls in enumerate(clf.classes_):
            top = np.argsort(clf.coef_[ci])[::-1][:20]
            fh.write(f"-- {cls}: " + ", ".join(f"{feats[i]} ({clf.coef_[ci][i]:.2f})" for i in top) + "\n")
        fh.write("\n")
print("\nwrote results to", OUT)
