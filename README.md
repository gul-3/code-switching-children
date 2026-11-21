# Code-switching in bilingual children's speech

Course project for Cognitive Models of Language Learning. Team: Gül Karabaş, Violeta Sandu, Sofia Sinisalmi.

Can machine learning identify code-switching in the speech of Spanish-English bilingual children? We classify each child utterance as `same` (in the language of the context), `other` (whole utterance in the other language) or `mixed` (both languages in one utterance), and compare lexical and grammatical features.

## Contents

| Path | What it is |
|---|---|
| `docs/FINDINGS.md` | All findings, numbered, each with the file it comes from |
| `docs/METHODS.md` | How the data were prepared and how the models were trained and evaluated |
| `results/RESULTS.md` | Model results and breakdowns |
| `results/ERROR_ANALYSIS.md` | Error analysis of the mixed class |
| `results/` | Result tables (CSV) |
| `scripts/` | Code, in the order below |

## Running it

The corpora are not included. Download Hoff, FerFuLice, Deuchar and Pérez-Bazán from the CHILDES Bilingual collection on TalkBank (free account needed) and unzip them into `data/` so that the folders are `data/Hoff/`, `data/FerFuLice/`, `data/Deuchar/` and `data/Perez/`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python scripts/build_dataset.py       # raw CHAT files -> data/all_corpora.csv
.venv/bin/python scripts/make_split.py          # -> data/folds.csv (5 folds grouped by child)
.venv/bin/python scripts/tag_pos.py             # -> data/all_corpora_tagged.csv (POS tags)
.venv/bin/python scripts/run_models.py          # -> results/ (baselines and models)
.venv/bin/python scripts/error_analysis.py      # -> results/error_analysis/
.venv/bin/python scripts/direction_analysis.py  # -> results/direction_analysis/
```

`build_dataset.py` and `make_split.py` were originally run with pandas 3.0.2 and scikit-learn 1.8.0; the other scripts with the versions in `requirements.txt`.

Files containing transcript text (the dataset, `results/oof_predictions.csv` and the example tables of the error analysis) are left out of the repository and are recreated by the scripts.
