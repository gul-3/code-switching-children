# Results

Produced by `scripts/run_models.py` (setup in `docs/METHODS.md` section 11). Main sample: ages up to 3;6, 121,480 child utterances, 5-fold cross-validation grouped by child, 3 classes, logistic regression with balanced class weights and C = 1.0 (not tuned).

## 1. Main result (5-fold CV, mean ± SD)

| Model | Macro-F1 | F1 same | F1 other | F1 mixed | Accuracy |
|---|---|---|---|---|---|
| B0 majority class | 0.311 ± 0.002 | 0.932 | 0.000 | 0.000 | 0.873 |
| B1 length only | 0.268 ± 0.016 | 0.706 | 0.000 | 0.098 | 0.525 |
| B2 context only | 0.189 ± 0.079 | 0.467 | 0.046 | 0.054 | 0.329 |
| M1 lexical | 0.725 ± 0.033 | 0.964 | 0.808 | 0.402 | 0.930 |
| M2a grammatical | 0.543 ± 0.023 | 0.860 | 0.569 | 0.200 | 0.767 |
| M2b POS sequence only | 0.508 ± 0.014 | 0.834 | 0.506 | 0.184 | 0.731 |
| M3 lexical + grammatical | 0.735 ± 0.050 | 0.959 | 0.792 | 0.452 | 0.923 |

Per-fold values: `cv_folds.csv`. Summed confusion matrices: `confusion_<model>.csv`.

B1 and B2 score below B0. With balanced class weights they are pushed towards the rare classes, but length or context alone cannot tell which utterances are switches, so they make many wrong `other` and `mixed` predictions.

Confusion matrix for M3, summed over the test folds:

| | pred same | pred other | pred mixed |
|---|---|---|---|
| true same | 98,958 | 4,372 | 2,819 |
| true other | 706 | 11,361 | 452 |
| true mixed | 519 | 531 | 1,762 |

## 2. Research questions (preliminary)

RQ1, which features predict switching: words, read together with the context language. The largest M1 coefficients (`top_features.txt`) are, for `other`, Spanish words in English sessions (`eng§este`, `eng§mira`, `eng§aquí`, `eng§agua`) and English words in Spanish sessions (`spa§the`, `spa§it`, `spa§fish`); for `mixed`, Spanish function words (`un`, `el`, `una`, `es`) and words such as `abuela` and `papilla`. The function-word pattern is followed up in the error analysis (determiner + noun from the other language, e.g. "un shark", "this gomita").

RQ2, lexical vs grammatical: grammatical features alone (M2a 0.543, M2b 0.508) are well above the baselines but below words (M1 0.725). Adding grammar to words (M3) adds 0.010 macro-F1, mostly in the mixed class (0.402 to 0.452), which is less than the fold-to-fold SD. The strongest M2b features are still cues to language identity: `eng§es:PROPN` points to `same` and `spa§es:PROPN` to `other`, because the Spanish tagger labelling a word as a proper noun usually means the word is English. So part of what the grammatical models learn is which language the words are in (METHODS section 10).

RQ3, generalisation: all scores above are on children the model did not see. Trained on Hoff, Deuchar and Pérez-Bazán and tested on FerFuLice up to 3;6 (`loco_ferfulice.csv`), M1 reaches 0.635 (other 0.724, mixed 0.205). This is lower than on Hoff (0.766) but far above the baselines, and the transfer works mainly for full switches.

| Model (trained without FerFuLice, tested on it) | Macro-F1 | F1 other | F1 mixed |
|---|---|---|---|
| M1 lexical | 0.635 | 0.724 | 0.205 |
| M2a grammatical | 0.462 | 0.410 | 0.119 |
| M2b POS sequence | 0.452 | 0.396 | 0.109 |
| M3 lexical + grammatical | 0.600 | 0.624 | 0.224 |

The twins are in fold 0 of the CV, so their CV score is also a "trained without FerFuLice" score, which is why both give 0.635 for M1.

## 3. Breakdowns (out-of-fold predictions)

By corpus (macro-F1):

| Corpus | M1 | M2a | M2b | M3 | M1 F1 other | M1 F1 mixed |
|---|---|---|---|---|---|---|
| Hoff | 0.766 | 0.578 | 0.536 | 0.793 | 0.862 | 0.465 |
| FerFuLice ≤ 3;6 | 0.635 | 0.464 | 0.452 | 0.601 | 0.724 | 0.204 |
| Deuchar | 0.596 | 0.424 | 0.382 | 0.562 | 0.644 | 0.186 |
| Pérez-Bazán | 0.314 | 0.221 | 0.229 | 0.286 | 0.123 | 0.055 |

Pérez-Bazán is very low. It has very few switches (131 other, 17 mixed), its context is only the home language (METHODS section 3), and the error analysis suggests some switches there are not marked.

By age (macro-F1):

| Age | Utterances | % other | % mixed | M1 | M2b | M3 |
|---|---|---|---|---|---|---|
| < 2;0 | 5,212 | 9.7 | 1.9 | 0.579 | 0.408 | 0.534 |
| 2;0 to 2;5 | 9,134 | 7.8 | 2.5 | 0.539 | 0.420 | 0.545 |
| 2;6 to 2;11 | 44,813 | 11.8 | 2.0 | 0.735 | 0.505 | 0.748 |
| 3;0 to 3;6 | 62,321 | 9.6 | 2.6 | 0.757 | 0.533 | 0.767 |

Ages below 2;6 come only from FerFuLice, Deuchar and Pérez-Bazán, so the lower scores there cannot be put down to age (METHODS section 9). Within 2;6 to 3;6, where Hoff dominates, scores rise slightly with age.

By length (`by_length.csv`): mixed F1 is 0 for one-word utterances, since almost none are mixed. For 2 to 5 words M1 reaches mixed F1 of about 0.44 to 0.49, dropping to 0.36 for 6+ words.

## 4. Correction made during the first run

The first run (`results/run1_no_context_interaction/`) gave words and context to the models as separate features. M1 reached only 0.421 macro-F1 (F1 other 0.28). A linear model cannot learn that the same word is `same` in one context and `other` in another ("mira" in a Spanish vs an English session); that needs an interaction between word and context. In the corrected run each linguistic feature appears twice, once as is and once prefixed with the session language (`eng§mira`, `spa§mira`). Nothing else changed, and the baselines are identical in both runs.

## 5. Open points

- No hyperparameter tuning (C fixed at 1.0). Tuning would need a nested, grouped CV.
- Error analysis of the mixed class: done, see `ERROR_ANALYSIS.md`.
- Descriptive analysis of the FerFuLice twins from 1 to 6 years: not done.

## 6. Analysis by switch direction

Produced by `scripts/direction_analysis.py` (METHODS section 14); tables in `results/direction_analysis/`. Counts and proportions only, no significance tests.

Switching by session language (% of child utterances):

| Corpus | Session | n | % same | % other | % mixed |
|---|---|---|---|---|---|
| Hoff | eng | 43,082 | 87.5 | 10.9 | 1.6 |
| Hoff | spa | 48,841 | 83.8 | 13.0 | 3.3 |
| FerFuLice | eng | 12,576 | 89.0 | 7.9 | 3.1 |
| FerFuLice | spa | 10,073 | 97.7 | 1.5 | 0.9 |
| Deuchar | eng | 1,276 | 90.4 | 8.1 | 1.5 |
| Deuchar | spa | 1,533 | 91.5 | 6.8 | 1.8 |
| Pérez-Bazán | spa | 4,099 | 96.4 | 3.2 | 0.4 |

In Hoff, switching and mixing are both more frequent in Spanish sessions; in FerFuLice both are more frequent in English sessions. In both corpora mixing is higher in the session type where full switches are also more frequent. Deuchar (one child) shows no clear difference.

F1 for `other` by direction (M1):

| Corpus | English session (child speaks Spanish) | Spanish session (child speaks English) |
|---|---|---|
| All | 0.873 | 0.756 |
| Hoff | 0.902 | 0.836 |
| FerFuLice | 0.772 | 0.420 |
| Deuchar | 0.746 | 0.533 |

Switches into Spanish are recognised better in every corpus. FerFuLice and Deuchar have few `other` utterances in Spanish sessions (148 and 104), which may explain part of their gap, but in Hoff the Spanish sessions have more `other` utterances (6,341 vs 4,697) and the gap is still there. For `mixed`, M3 is similar in both directions (0.459 vs 0.445).

Lexical diversity does not explain the direction gap. Spanish switches have 8.7 word types per 100 tokens and English switches 6.8 (Spanish switches are shorter: 10,649 tokens in 5,795 utterances vs 15,638 in 6,724), and 50% vs 44% of utterances contain one of that direction's 10 most frequent words. Those words are *sí, este, no, aquí, esto, yo, mira, es, y, el* for Spanish switches and *i, a, the, it, this, you, is, water, up, go* for English switches.

Preceding adult utterance: 93.5% of child `other` utterances and 92.7% of `mixed` utterances follow an adult utterance in the session language (98.2% for `same`). Only 471 of 12,519 `other` utterances (Hoff 394 of 11,038) follow an adult utterance in the other language.

Determiner pattern by session (mixed utterances): Spanish determiner + English word outnumbers the reverse in both Hoff session types (Spanish sessions 538 vs 144, English sessions 160 vs 103) and in FerFuLice English sessions (33 vs 20). The difference is largest in Hoff Spanish sessions.
