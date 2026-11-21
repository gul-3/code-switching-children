"""Add part-of-speech / morphology tags to data/all_corpora.csv for Model 2.

Every utterance is tagged TWICE, with an English and a Spanish spaCy model, because
we cannot know a word's language without using the (leaky) annotation. The model
gets both views and the context language.

- Input text: the `clean` column, already split into words on whitespace. We pass
  these words to spaCy as given (no re-tokenisation), so every word gets exactly
  one tag in each column and the columns stay aligned with `clean`.
- Only the components needed for POS and morphology run (tok2vec, tagger /
  morphologizer, attribute_ruler). Parser, NER and lemmatiser are disabled.
- Output format per word: word|UPOS|Feat1=Val|Feat2=Val (or `_` if no features).

Usage (inside the project venv):
    .venv/bin/python scripts/tag_pos.py
Writes data/all_corpora_tagged.csv (all_corpora.csv + tags_en, tags_es).
"""
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = {"tags_en": "en_core_web_sm", "tags_es": "es_core_news_sm"}
KEEP = {"tok2vec", "tagger", "morphologizer", "attribute_ruler"}


def tag_column(texts, model_name):
    nlp = spacy.load(model_name)
    disabled = [p for p in nlp.pipe_names if p not in KEEP]
    docs = (Doc(nlp.vocab, words=t.split()) for t in texts)
    out = []
    with nlp.select_pipes(disable=disabled):
        for doc in nlp.pipe(docs, batch_size=2000):
            out.append(" ".join(f"{tok.text}|{tok.pos_}|{str(tok.morph) or '_'}" for tok in doc))
    return out, f"{model_name} {nlp.meta['version']} (spaCy {spacy.__version__})"


def main():
    df = pd.read_csv(DATA / "all_corpora.csv", dtype=str, keep_default_na=False)
    for col, name in MODELS.items():
        df[col], version = tag_column(df["clean"].tolist(), name)
        print(f"{col}: {version}")
    df.to_csv(DATA / "all_corpora_tagged.csv", index=False)
    print("wrote", DATA / "all_corpora_tagged.csv", len(df), "rows")


if __name__ == "__main__":
    main()
