# Findings

This file lists what the project found. Every number here comes from the scripts in `scripts/` and the tables in `results/`; the source of each finding is given in brackets. No literature is cited and no outside knowledge is used. Linking the findings to the literature is left for the report, and any such link has to be checked against the original papers.

Sample: 121,480 child utterances up to age 3;6 from four CHILDES corpora (Hoff, FerFuLice, Deuchar, Pérez-Bazán). Three classes: `same` (context language), `other` (whole utterance in the other language, inter-sentential), `mixed` (both languages in one utterance, intra-sentential). Model scores are out-of-fold, i.e. always on children the model was not trained on.

No significance tests were run. Differences are described, not tested.

## A. What the children do (descriptive)

### 1. Children mostly speak the context language

87.4% of child utterances are in the context language, 10.3% are full switches and 2.3% are mixed. Across age bands mixed utterances stay between 1.9% and 2.6% and full switches between 7.8% and 11.8%.
[`data/all_corpora.csv`; `results/by_age.csv`]

### 2. Switches mostly happen while the adult stays in the context language

93.5% of the children's `other` utterances and 92.7% of their `mixed` utterances follow an adult utterance in the session language (98.2% for `same`). Only 471 of 12,519 `other` utterances follow an adult utterance in the other language (Hoff: 394 of 11,038). The context-only baseline, which knows the session language and the language of the preceding adult utterance, cannot predict switching (macro-F1 0.189).
[`results/direction_analysis/prev_adult_*.csv`, `other_after_adult_in_other_language.csv`; `results/cv_summary.csv`]

### 3. The amount of switching depends on the session language, in opposite directions in the two largest corpora

| Corpus | Session | % other | % mixed |
|---|---|---|---|
| Hoff | English | 10.9 | 1.6 |
| Hoff | Spanish | 13.0 | 3.3 |
| FerFuLice | English | 7.9 | 3.1 |
| FerFuLice | Spanish | 1.5 | 0.9 |
| Deuchar | English | 8.1 | 1.5 |
| Deuchar | Spanish | 6.8 | 1.8 |

In Hoff the children switch and mix more in Spanish sessions; in FerFuLice (the twins) they do so in English sessions. In both corpora, mixing is higher in the session type where full switches are also more frequent. Deuchar (one child) shows no clear difference. The data contain no measure of the children's language dominance, so the reason for this pattern cannot be tested here.
[`results/direction_analysis/labels_by_corpus_session_pct.csv`]

### 4. Most mixed utterances contain a single word from the other language

Using a word-language lexicon built from the monolingual utterances (a heuristic, see METHODS section 13), 1,553 of 2,812 mixed utterances (55%) contain exactly one other-language word. Only 235 (8%) contain two or more words of each language.
[`results/error_analysis/mixed_by_composition.csv`]

### 5. Spanish determiner + English word is much more frequent than the reverse

Within mixed utterances, a Spanish determiner directly followed by an English word (*un cake*, *una zebra*) occurs 750 times; an English determiner followed by a Spanish word (*a jirafa*, *the sopa*) occurs 271 times. The direction holds in Hoff Spanish sessions (538 vs 144), Hoff English sessions (160 vs 103) and FerFuLice English sessions (33 vs 20). The determiner lists are fixed in `error_analysis.py`, and the language of the following word comes from the heuristic lexicon.
[`results/error_analysis/determiner_pattern_by_corpus.csv`; `results/direction_analysis/determiner_pattern_by_corpus_session.csv`]

## B. What the models show

### 6. Switching can be predicted from the words, in children the model has not seen

| Model | Macro-F1 (mean ± SD, 5 folds) |
|---|---|
| Majority / length / context baselines | 0.311 / 0.268 / 0.189 |
| M1 words | 0.725 ± 0.033 |
| M2a POS + morphology | 0.543 ± 0.023 |
| M2b POS sequence only | 0.508 ± 0.014 |
| M3 words + grammar | 0.735 ± 0.050 |

Length and context alone do not predict switching. Words do, and the result holds for children outside the training data.
[`results/cv_summary.csv`]

### 7. A word only signals a switch in combination with the context language

When words and context were given to the model as separate features, M1 reached 0.421. When each word was also given together with the session language (`eng§mira`, `spa§mira`), it reached 0.725. Part of this is built into the task, because `other` and `same` are defined relative to the session language.
[`results/run1_no_context_interaction/cv_summary.csv`; `results/cv_summary.csv`]

### 8. Words carry more information than our grammatical features; grammar adds a little for mixed utterances

Grammatical features alone are well above the baselines but below words. Adding them to words raises macro-F1 by 0.010 and mixed F1 from 0.402 to 0.452, which is smaller than the fold-to-fold SD. The tags carry some language information themselves (the Spanish tagger tends to label English words as proper nouns and vice versa), so the grammatical models are not purely grammatical.
[`results/cv_summary.csv`; `results/top_features.txt`; METHODS section 10]

### 9. The strongest cues

From the largest coefficients of M1:
- `other`: Spanish words in English sessions (*este, un, esto, mira, aquí, sí, el, ese, mami, eso, y, yo, agua*) and English words in Spanish sessions (*the, it, fish*).
- `mixed`: Spanish function words (*un, es, el, una, otro*) and the words *abuela, papilla, mami, quiero*.

The `mixed` cues are consistent with finding 5.
[`results/top_features.txt`]

### 10. Full switches are easy to detect, mixed utterances are not

F1 for `other` is 0.81 (M1) and for `mixed` 0.40 to 0.45. For M3, mixed recall is 0.63 and precision 0.35. 90% of the false `mixed` predictions are ordinary monolingual utterances, which follows from the balanced class weights.
[`results/cv_summary.csv`; `results/ERROR_ANALYSIS.md`]

### 11. The model finds mixing when both languages are clearly present

| Mixed utterance contains | Found as mixed | Predicted other | Predicted same |
|---|---|---|---|
| 2+ words of each language | 0.88 | 0.05 | 0.07 |
| one other-language word | 0.68 | 0.10 | 0.22 |
| one context-language word | 0.63 | 0.32 | 0.05 |
| no clear context-language word | 0.27 | 0.61 | 0.12 |

Mixed utterances dominated by one language are read as that language. Since single-word insertions are the most common type (finding 4), they are the main source of missed mixed utterances.
[`results/error_analysis/mixed_by_composition_share.csv`]

### 12. The frequent determiner pattern is recognised much better

Spanish determiner + English word is found as mixed in 88% of cases, English determiner + Spanish word in 52%.
[`results/error_analysis/mixed_by_determiner_pattern_share.csv`]

### 13. Switches into Spanish are recognised better than switches into English

| Corpus | F1 other, English session (child speaks Spanish) | F1 other, Spanish session (child speaks English) |
|---|---|---|
| All | 0.873 | 0.756 |
| Hoff | 0.902 | 0.836 |
| FerFuLice | 0.772 | 0.420 |
| Deuchar | 0.746 | 0.533 |

This holds in every corpus. In FerFuLice and Deuchar the Spanish sessions have few `other` utterances (148 and 104), but in Hoff they have more than the English sessions and the gap remains. We checked lexical diversity as a possible explanation and it does not explain the gap (Spanish switches: 8.7 word types per 100 tokens; English switches: 6.8).
[`results/direction_analysis/scores_by_direction.csv`, `other_lexical_diversity.csv`]

### 14. The patterns transfer to another corpus, but mostly for full switches

Trained without FerFuLice and tested on it, M1 reaches macro-F1 0.635 (other 0.724, mixed 0.205). By corpus, M1 scores 0.766 on Hoff, 0.596 on Deuchar and 0.314 on Pérez-Bazán.
[`results/loco_ferfulice.csv`; `results/by_corpus.csv`]

### 15. Age and length

Within 2;6 to 3;6, where Hoff provides most of the data, M1 rises slightly with age (0.735 to 0.757). Below 2;6 the scores are lower (0.54 to 0.58), but these ages come only from FerFuLice, Deuchar and Pérez-Bazán, so age and corpus cannot be separated. One-word mixed utterances are never found; for 2 to 5 words M1 reaches mixed F1 0.44 to 0.49, and 0.36 for 6+ words.
[`results/by_age.csv`; `results/by_length.csv`]

## C. Findings about the data

### 16. Some errors are annotation gaps

In Pérez-Bazán, 181 of the 434 `same` utterances predicted as mixed contain English words (*apple juice*, *I want m my foca*). They look like switches that were not marked in the transcript, so the Pérez-Bazán scores are probably too low. In Hoff, 128 text and context combinations carry more than one label.
[`results/error_analysis/false_mixed_by_corpus.csv`; METHODS section 9]

### 17. The original `%mor` tier cannot be used for mixed utterances

TalkBank's `%mor` tier is generated automatically with one language per file. In Hoff, 90% of mixed utterances carry `-Nom` and 35% of their tags are proper nouns (3% elsewhere). We therefore re-tagged all utterances with both an English and a Spanish tagger.
[METHODS sections 9 and 10]

## What the project does not show

- Whether any difference is statistically significant (no tests were run).
- A developmental trend: age is confounded with corpus below 2;6, and the twins' data above 3;6 were not analysed.
- Why children switch more in one session language than the other: there is no dominance or proficiency measure in the data.
- How the children process language: the models show which cues are associated with switching, not the mechanism behind it.
- The word-language lexicon used in findings 4, 5, 11 and 12 is a heuristic. It misclassifies rare words, child forms and loanwords.
