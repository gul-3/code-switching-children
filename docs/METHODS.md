# Methods

This file records how the dataset was built and how the models were trained and evaluated. Results are in `results/RESULTS.md`, error analysis in `results/ERROR_ANALYSIS.md`, and a summary of all findings in `docs/FINDINGS.md`.

Pipeline:

```bash
python3 scripts/build_dataset.py                # raw CHAT files -> data/all_corpora.csv
python3 scripts/make_split.py                   # main sample -> data/folds.csv
.venv/bin/python scripts/tag_pos.py             # -> data/all_corpora_tagged.csv
.venv/bin/python scripts/run_models.py          # -> results/
.venv/bin/python scripts/error_analysis.py      # -> results/error_analysis/
.venv/bin/python scripts/direction_analysis.py  # -> results/direction_analysis/
```

## 1. Corpora

All corpora are from the CHILDES Bilingual collection on TalkBank. Downloading them requires a free TalkBank account, and each corpus has to be cited as TalkBank asks.

| Corpus | Used | Children | Ages | Setting | Citation |
|---|---|---|---|---|---|
| Hoff | yes | 82 | 2;6, 3;0, 3;6 (three waves) | Florida, USA. Caregiver asked to use only Spanish or only English in a session | Tulloch & Hoff (2022), *J. Child Language*, or another reference from the corpus page |
| FerFuLice | yes | 2 (twins) | 1;01 to 6;10 | Valladolid, Spain. Mother English, father Spanish | Liceras et al. (2008), *Lingua* 118(6) |
| Deuchar | yes | 1 | 1;03 to 2;06 | UK. Father Spanish, grandmother English | Deuchar & Quay (2000), OUP |
| Pérez-Bazán | yes | 6 | 1;02 to 3;05 | Michigan, USA. No language instructions | Pérez-Bazán (2002), PhD thesis, U. Michigan |
| Ticio | no | 1 | 1;x | Texas, USA | |

Excluded material:
- Ticio has no code-switching annotation and the child is mostly babbling.
- Hoff `mono/` contains English monolingual children.
- FerFuLice `biling/` contains four elicitation tasks (translation and code-switching tests) rather than spontaneous speech.

## 2. Unit of analysis

One target-child utterance (one CHAT main tier spoken by the target child). Adult utterances are not classified.

In Hoff, Deuchar and Pérez-Bazán the target child is `CHI`; in Pérez-Bazán the child is identified by the folder name. In FerFuLice the twins are `CHI` and `CHI2`, and `SOL` marks utterances where the transcribers could not tell which twin spoke. `SOL` is kept under its own id.

## 3. Context language

A switch is defined relative to the language of the interaction.

| Corpus | Source of the context language |
|---|---|
| Hoff | Folder `biling-eng` / `biling-spa`, the language the caregiver was asked to speak |
| FerFuLice | Folder `eng/` / `spa/`: English interlocutor (mother, English investigator) or Spanish interlocutor (father, Spanish investigator) |
| Deuchar | Folder `Eng/` (grandmother) / `Span/` (father) |
| Pérez-Bazán | No instructions were given. All files list Spanish first (`@Languages: spa, eng`), so the context is taken to be Spanish, the home language |

### 3.1 Language of the preceding adult utterance

As a turn-level context we also store `prev_adult_language`: the language of the most recent non-empty utterance by any non-target speaker before the child utterance, within the same file. Values are `eng`, `spa`, `mix`, or `none` if the child speaks first. Adult utterances are annotated in the same way as the children's, so the same rules (section 4) give their language. This uses the adult's annotation, not the child's, so it does not leak the label.

The preceding adult utterance is in the context language for about 98% of child utterances (Hoff 89,815 of 91,827; FerFuLice 45,489 of 46,307). In Hoff, 666 of the 11,038 child `other` utterances follow an adult utterance that is in the other language (394) or mixed (272). An earlier version of this file counted both groups as "other language"; the split is produced by `scripts/direction_analysis.py`. So children mostly switch while the adult stays in the context language. We therefore keep `other` defined relative to the session language and give `prev_adult_language` to the models as an extra context feature.

## 4. Labels

| label3 | Meaning | Rule |
|---|---|---|
| `mixed` | Both languages in one utterance (intra-sentential) | Hoff precode `[- hun]`, or any word marked `@s` (FerFuLice, Deuchar, Pérez-Bazán) |
| `other` | Whole utterance in the other language (inter-sentential) | Language precode (`[- eng]` / `[- spa]`) different from the context language |
| `same` | Utterance in the context language | No precode, or a precode equal to the context language |

If more than one rule applies, `mixed` wins over `other`, and `other` over `same`.

The Hoff corpus page says that `[- hun]` marks mixed utterances and does not mean Hungarian. It also mentions an "unassigned" code `[-una]`, but no child utterance in the files has it. For `@s`, only the exact code (optionally `@s:eng`) counts; `@si` (singing), `@c` (child-invented word), `@l` (letter), `@fp` (filled pause), `@i` (interjection) and `@x` do not.

## 5. Cleaning (`clean` column)

The aim is to keep the words the child said without transcription codes. Steps, in order:

| Step | Rule | Example |
|---|---|---|
| 1 | Remove media time stamps | `yeah . •5231_5731•` → `yeah` |
| 2 | Remove all `[...]` annotations (precodes, `[: x]`, `[/]`, `[*]`, overlaps, comments) | `me [: my] teeth .` → `me teeth` |
| 3 | Remove `<` `>` | `<be ni(ce)> [//] (o)kay .` → `be nice okay` |
| 4 | Drop non-word tokens: `&=laugh`, `&-um`, `&~frag`, `xxx`/`yyy`/`www`, `0x`, `+/.`, `+<`, `(.)`, punctuation | `&=laugh green .` → `green` |
| 5 | Inside words: strip `@` codes, keep letters inside `( )`, turn `_` and `+` into spaces, remove `:` and other prosodic marks | `whoa@i` → `whoa`, `oop(s)` → `oops`, `french_fries` → `french fries`, `le:che` → `leche` |

We keep what the child said rather than the target form in `[: x]` (`it broked [: broke]` → `it broked`), and repetitions stay as spoken. Utterances that are empty after cleaning (only unintelligible speech, laughter or pauses) are dropped.

| Corpus | Child utterances | Dropped | Kept |
|---|---|---|---|
| Hoff | 113,028 | 21,105 | 91,923 |
| FerFuLice | 48,006 | 1,210 | 46,796 |
| Deuchar | 4,042 | 1,233 | 2,809 |
| Pérez-Bazán | 4,308 | 209 | 4,099 |

## 6. Check against an independent preprocessing of Hoff

One team member had preprocessed Hoff separately (`hoff_retagged.csv`). We compared the two row by row, matching on session and line number.

- Labels agree on all 91,923 shared rows.
- 74 rows exist only in the other file. All contain only unintelligible speech (e.g. `+< xxx@x .`), which that cleaning kept as `xxx`.
- 238 rows differ in the cleaned text because of small cleaning differences: lengthening colons (`le che` vs `leche`), `xxx` inside `< >`, and `&~` fragments.

## 7. Final dataset (`data/all_corpora.csv`)

### 7.1 Main sample (ages up to 3;6)

All models use only utterances with `analysis_age_months` ≤ 42 (`main_sample == 1`). Reasons for the cut-off:
- Everything above 3;6 comes from the two FerFuLice twins, so age and corpus cannot be separated there.
- Switching is much lower in that older data (mixed 0.5 to 0.8%, other about 0.9%), and we cannot tell whether this is age or these two children.
- Training mostly on 2 to 3 year olds and testing on 6 year olds would not be a useful test.

`analysis_age_months` is the wave for Hoff (30/36/42) and the `@ID` age for the other corpora, giving a range of 13 to 42 months.

| Corpus | Main sample | same | other | mixed |
|---|---|---|---|---|
| Hoff | 91,923 | 78,608 | 11,038 | 2,277 |
| FerFuLice (≤ 3;6) | 22,649 | 21,034 | 1,143 | 472 |
| Deuchar | 2,809 | 2,556 | 207 | 46 |
| Pérez-Bazán | 4,099 | 3,951 | 131 | 17 |
| Total | 121,480 | 106,149 | 12,519 | 2,812 |

The 24,147 FerFuLice utterances above 3;6 stay in the file with `main_sample == 0`. They are not used for training or testing.

Utterances by age band (all data):

| Age | Deuchar | FerFuLice | Hoff | Pérez-Bazán |
|---|---|---|---|---|
| < 2;0 | 1,312 | 3,699 | 0 | 201 |
| 2;0 to 2;5 | 1,197 | 5,594 | 0 | 2,343 |
| 2;6 to 3;6 | 300 | 13,356 | 91,923 | 1,555 |
| 3;7 to 4;11 | 0 | 16,047 | 0 | 0 |
| 5;0+ | 0 | 8,100 | 0 | 0 |

### 7.2 All data by context

| Corpus | Context | same | other | mixed |
|---|---|---|---|---|
| Hoff | eng | 37,701 | 4,697 | 684 |
| Hoff | spa | 40,907 | 6,341 | 1,593 |
| FerFuLice | eng | 23,951 | 1,186 | 486 |
| FerFuLice | spa | 20,860 | 181 | 132 |
| Deuchar | eng | 1,154 | 103 | 19 |
| Deuchar | spa | 1,402 | 104 | 27 |
| Pérez-Bazán | spa | 3,951 | 131 | 17 |
| Total (145,627) | | 129,926 | 12,743 | 2,958 |

### 7.3 Columns

| Column | Use |
|---|---|
| `clean`, `tags_en`, `tags_es`, `session_language`, `prev_adult_language`, `analysis_age_months`, `n_tokens` | Model inputs |
| `child_id`, `corpus`, `main_sample` | Grouping, filtering and reporting only |
| `label3` | Target |
| `utterance_raw`, `utt_language`, `label_rule` | Never used as inputs, because they contain the codes the label is derived from. Kept so labels can be checked |

`age_months` comes from the `@ID` header. For Hoff we use `age_wave_months` (30/36/42) from the folder, because the header age is missing in 283 rows and sometimes disagrees with the wave (e.g. child 028).

## 8. Cross-validation folds (`data/folds.csv`)

The main sample is split into 5 folds grouped by child. Each model is trained 5 times, each time tested on one fold and trained on the other four, so every child is tested once by a model that never saw that child. We report the mean and SD over the 5 test folds.

Grouping by child tests generalisation to new children and stops the model from memorising repeated utterances (73% of cleaned Hoff utterances are repeats). The FerFuLice twins (`CHI`, `CHI2`, `SOL`) form one group because they get nearly the same input. Folds are made with `StratifiedGroupKFold` (5 folds, shuffled, seed 42) so class proportions are similar.

| Fold | Utterances | Groups | same | other | mixed |
|---|---|---|---|---|---|
| 0 | 28,760 | 7 | 89.5% | 8.5% | 2.0% |
| 1 | 23,250 | 19 | 86.6% | 11.0% | 2.4% |
| 2 | 23,013 | 20 | 86.6% | 11.0% | 2.4% |
| 3 | 23,409 | 22 | 87.0% | 10.6% | 2.4% |
| 4 | 23,048 | 22 | 86.7% | 10.9% | 2.4% |

Fold 0 holds the twins (22,649 utterances) and 6 Hoff children. Deuchar is in fold 1 and the Pérez-Bazán children are in folds 2 to 4. An earlier version used a single 80/20 split; it was replaced by these folds after the age restriction.

## 9. Limitations

1. Annotation schemes differ. Hoff codes whole utterances; the other corpora code single words (`@s`) plus a precode for whole-utterance switches. Mapping both to three classes is our decision.
2. The context is less controlled outside Hoff. In FerFuLice English sessions the Spanish-speaking father is sometimes present, and in Pérez-Bazán the context is just the home language.
3. Some switches are not marked. A rough word-list check found a few hundred FerFuLice child utterances (<1%) with unmarked other-language words.
4. Hoff annotation is not fully consistent: 128 text and context combinations have more than one label (e.g. "es" in English sessions is `other` 3 times and `same` 7 times).
5. Classes are imbalanced (mixed is 2.3% of the main sample), so we report macro-F1 and per-class F1.
6. Corpus and age are confounded. Ages above 3;6 come only from FerFuLice (excluded, section 7.1); ages below 2;6 only from FerFuLice, Deuchar and Pérez-Bazán. Age comparisons are safest within 2;6 to 3;6.
7. Given the words and the context, `other` is close to language identification and expected to be easy. `mixed` is the more informative class.
8. The original `%mor` tier cannot be used for mixed utterances. TalkBank generates it automatically with one language per file (`engine=stanza ... lang=spa`). In Hoff, 90% of mixed utterances carry `-Nom` and 35% of their tags are proper nouns, against 3% elsewhere. POS features are therefore produced by re-tagging `clean` (section 10).

## 10. POS and morphology tagging (`data/all_corpora_tagged.csv`)

Taggers: spaCy 3.8.20 with `en_core_web_sm` 3.8.0 and `es_core_news_sm` 3.8.0. Both give Universal Dependencies POS tags and morphological features, and all corpora are tagged with the same models.

- Every utterance is tagged twice, with the English model (`tags_en`) and the Spanish model (`tags_es`). Picking one model per utterance would require knowing its language, which is the label we want to predict.
- The input is `clean` split on whitespace, without re-tokenisation, so each word gets one tag per column (checked on all rows). Contractions like `it's` stay one token.
- Only tok2vec, tagger/morphologizer and attribute_ruler run; parser, NER and lemmatiser are off.
- Format per word: `word|UPOS|Feature=Value|...`, or `_` if there are no features.

Checks: when the tagger matches the utterance language, the tags look normal (proper nouns are 3.9% of English tags in English contexts and 7.2% of Spanish tags in Spanish contexts, for multi-word `same` utterances). When it does not match, it tends to tag foreign words as `PROPN`; over all utterances, 26% of the English tags are `PROPN`. The tags therefore still carry some information about language, which matters when interpreting Model 2. The POS tags in the other team member's Hoff file agree with ours on only about 61% of tokens (different tagger models, e.g. theirs tagged Spanish words as `X`). We use our tags for all corpora so tagging is consistent.

## 11. Models and evaluation

Code: `scripts/run_models.py`. All models use the main sample and the 5 folds, and all receive the context features (`session_language`, `prev_adult_language`).

| Model | Input | Question |
|---|---|---|
| B0 majority | nothing, always `same` | Floor |
| B1 length | number of words | Is it just length? |
| B2 context | context features only | Is it just context? |
| M1 lexical | words + context | How much do words predict? |
| M2a grammatical | UPOS + morphology from both taggers + context | How much does grammar predict? |
| M2b POS sequence | UPOS n-grams only + context | Grammar in its simplest form, given that tags partly encode language |
| M3 lexical + grammatical | M1 + M2a features | Does grammar add to words? |

Classifier: `LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000)`. It was chosen because its coefficients can be read. Balanced weights give each class weight inversely proportional to its frequency so that `mixed` is not ignored. C is fixed and not tuned, so no test fold affects any setting. B0 is `DummyClassifier(strategy="most_frequent")`.

Features:
- Words (M1, M3): lowercased unigrams and bigrams of `clean`.
- Grammar (M2a, M3): for each tagger (`en:` / `es:`), the UPOS of each word, UPOS plus each morphological feature, and UPOS bigrams with start and end markers.
- POS sequence (M2b): UPOS uni-, bi- and trigrams per tagger.
- These are weighted with TF-IDF (sublinear tf, `min_df=2`), fitted on the training folds only.
- Context: one-hot `session_language` and `prev_adult_language`.
- Length (B1): one-hot number of words (1 to 5, 6+).

Each linguistic feature appears twice: as is, and prefixed with the session language (`mira` and `eng§mira`). The label depends on the word together with the context ("mira" is `same` in a Spanish session and `other` in an English one), and a linear model cannot learn that from separate word and context features. A first run without this (`results/run1_no_context_interaction/`) gave M1 macro-F1 0.421 (F1 other 0.28) against 0.725 with it.

`top_features.txt` lists the 20 largest coefficients per class for M1, M2a and M2b, from models refitted on the whole main sample. It is used for interpretation only.

Evaluation: macro-F1 (mean and SD over folds) and per-class F1, confusion matrices, and breakdowns by corpus, age and length. Generalisation across children comes from the grouped folds. Generalisation across corpora is tested by training without FerFuLice and testing on FerFuLice up to 3;6 (another country and family setup).

Not done: a descriptive analysis of the twins' switching from 1 to 6 years using the `main_sample == 0` data.

## 12. Software

- Python 3.13.
- System Python (pandas 3.0.2, scikit-learn 1.8.0) for `build_dataset.py` and `make_split.py`.
- Virtual environment `.venv` (spaCy 3.8.20, scikit-learn 1.9.1, pandas 3.1.1) for the other scripts. See `requirements.txt`.

## 13. Error analysis

Code: `scripts/error_analysis.py`. It looks at the out-of-fold predictions of M3 for the `mixed` class; results are in `results/ERROR_ANALYSIS.md`.

To describe errors, each word gets a language from how often it appears in monolingual (`same` / `other`) utterances: Spanish if more than 85% of its occurrences are Spanish, English if fewer than 15%, "both" otherwise, and "unknown" if seen in fewer than 3 utterances. This lexicon is only used to describe errors; no model uses it.

## 14. Analysis by switch direction

Code: `scripts/direction_analysis.py`. Descriptive only: no model is trained and no significance test is run. It reads the main sample and `results/oof_predictions.csv` and writes to `results/direction_analysis/`:

- label counts and percentages by corpus and session language;
- out-of-fold F1 of M1 and M3 for `other` and `mixed` per session language;
- the language of the preceding adult utterance for child `other` utterances;
- the determiner pattern of `mixed` utterances by corpus and session language (same lexicon as section 13);
- lexical diversity of `other` utterances per direction (types, tokens, types per 100 tokens, share of utterances containing one of the 10 most frequent words).

It imports `error_analysis.py` to reuse the lexicon, which re-runs that script and rewrites the same tables. Results are in `results/RESULTS.md` section 6.
