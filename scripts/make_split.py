"""Child-grouped cross-validation folds for the main sample of data/all_corpora.csv.

- Only the main sample is used (main_sample == 1: ages up to 3;6).
- A child's utterances are never in more than one fold, so every test fold
  contains only unseen children (no memorising repeated utterances of a child).
- FerFuLice twins (CHI, CHI2) and the ambiguous SOL ('Simon or Leo') utterances are
  one group: the twins share almost the same input, so splitting them would leak.
- StratifiedGroupKFold (5 folds, seed 42) keeps label3 proportions similar per fold.

Writes data/folds.csv (group, child_id, corpus, fold 0-4) and prints the composition.
Main results: 5-fold CV, train on 4 folds, test on the 5th, report mean ± SD.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_FOLDS = 5

df = pd.read_csv(DATA / "all_corpora.csv", dtype=str, keep_default_na=False)
df = df[df["main_sample"] == "1"].copy()
df["group"] = df["child_id"].where(df["corpus"] != "ferfulice", "ferfulice_twins")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["fold"] = -1
for k, (_, test_idx) in enumerate(sgkf.split(df, df["label3"], df["group"])):
    df.iloc[test_idx, df.columns.get_loc("fold")] = k

df[["group", "child_id", "corpus", "fold"]].drop_duplicates() \
    .sort_values(["fold", "group"]).to_csv(DATA / "folds.csv", index=False)

print("main sample utterances:", len(df))
print("\nutterances per fold:", df["fold"].value_counts().sort_index().to_dict())
print("groups per fold    :", df.drop_duplicates("group")["fold"].value_counts().sort_index().to_dict())
print("\nlabel3 share per fold")
print(pd.crosstab(df["fold"], df["label3"], normalize="index").round(3))
print("\nutterances by corpus and fold")
print(pd.crosstab(df["corpus"], df["fold"]))
