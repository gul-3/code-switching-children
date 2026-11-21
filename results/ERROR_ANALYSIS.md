# Error analysis: the mixed class

Produced by `scripts/error_analysis.py`; tables in `results/error_analysis/`. The utterance-level example files (`examples_mixed.csv`, `examples_false_mixed.csv`) are not in the repository because they contain transcript text; they are recreated by running the script.

We look at the out-of-fold predictions of M3 (lexical + grammatical), the best model for `mixed`. Every prediction comes from a model that had not seen that child.

| | Value |
|---|---|
| Mixed utterances | 2,812 |
| Found (true positives) | 1,762 |
| Recall | 0.627 |
| Precision | 0.350 |
| Missed, predicted `same` | 519 |
| Missed, predicted `other` | 531 |
| False alarms (`same` or `other` predicted as `mixed`) | 3,271 |

## Word-language lexicon

The corpora mark language per utterance, not per word. To describe what is inside a mixed utterance we give each word a language based on the monolingual utterances: Spanish if more than 85% of its occurrences are in Spanish utterances, English if fewer than 15%, "both" in between (e.g. *mama, papa, no*), and "unknown" if seen in fewer than 3 utterances. This is a heuristic used only to group errors; no model uses it and it does not change any score. It gets rare words, child forms (*stubo*, *friendsies*) and loanwords (*churro*) wrong.

## 1. Errors depend on how much of each language the utterance contains

| Composition of the mixed utterance | n | → mixed | → other | → same |
|---|---|---|---|---|
| E: 2+ words of each language | 235 | 0.88 | 0.05 | 0.07 |
| C: one other-language word | 1,553 | 0.68 | 0.10 | 0.22 |
| D: one context-language word | 407 | 0.63 | 0.32 | 0.05 |
| A: no clear other-language word | 304 | 0.52 | 0.14 | 0.34 |
| B: no clear context-language word | 313 | 0.27 | 0.61 | 0.12 |

The model mostly goes by the dominant language of the utterance. When both languages are well represented it finds 88% of the mixed utterances. With only one other-language word the utterance is often read as `same` (22%), e.g. *"creo que es un deer"* or *"los away"* in a Spanish session. When most words are in the other language it is often read as `other` (61% for group B), e.g. *"es una dinosaur"* in an English session.

Mixed utterances with a single other-language word are the most common type (1,553 of 2,812, 55%), so they are the main source of missed mixed utterances. Group A contains many words the lexicon cannot place, such as child forms, rare plurals (*reindeers*) and loanwords (*churro*); for these the switched word is probably hard for the model to recognise as well.

## 2. Spanish determiner + English word

We looked for a determiner of one language directly followed by a word of the other. Spanish determiners: *un, una, el, la, los, las, mi, este, esta, otro, otra*. English determiners: *a, the, this, my, that, another*.

| Pattern | n (% of mixed) | Found as mixed |
|---|---|---|
| Spanish determiner + English word (*un cake, una zebra, un big doggie*) | 750 (27%) | 0.88 |
| English determiner + Spanish word (*a jirafa, a oso, the sopa*) | 271 (10%) | 0.52 |
| Other mixed utterances | 1,791 (64%) | 0.54 |

Spanish determiner + English word is almost three times as frequent as the reverse in the children's speech. This holds in Hoff (698 vs 247) and in FerFuLice (38 vs 21), and in both Hoff session types (see `RESULTS.md` section 6). This is a property of the data, not of the model.

The model recognises the frequent direction much better (0.88 vs 0.52). Spanish determiners are frequent and language-specific, and *un, el, una* are among the strongest M1 features for `mixed` (`top_features.txt`). The reverse direction starts with *a* or *the*, which are also frequent in plain English utterances.

## 3. False alarms

`same` utterances predicted as `mixed` (2,819):

| Corpus | No other-language word | Words of both languages | Only other-language words | Total |
|---|---|---|---|---|
| Hoff | 1,467 | 35 | 8 | 1,510 |
| FerFuLice | 717 | 40 | 19 | 776 |
| Pérez-Bazán | 253 | 103 | 78 | 434 |
| Deuchar | 95 | 2 | 2 | 99 |
| Total | 2,532 | 180 | 107 | 2,819 |

Most false alarms (2,532, 90%) are ordinary monolingual utterances such as *"es un trompo"* or *"el tigre aquí"*. They follow from the balanced class weights, which make `mixed` as important as `same` and so trade precision (0.35) for recall (0.63). A different class weight or decision threshold, chosen with nested CV, would move this trade-off.

In Pérez-Bazán, 181 of the 434 false alarms contain English words, e.g. *"apple juice"*, *"need apple"*, *"I want m my foca"*, *"you do not have fresas"*. These look like English or mixed utterances in a Spanish home context that were not given a language code. In such cases the model is probably right and the label wrong, which fits limitation 3 in METHODS section 9 and helps explain the low Pérez-Bazán scores.

## 4. Length

One-word mixed utterances (14) are never found. For 2 to 6 words recall is fairly stable (0.61 to 0.68), so length is not the main problem within the mixed class; the dominant-language effect in section 1 is.

## Summary

1. The model finds mixing well when both languages are clearly present, and especially in the frequent Spanish determiner + English word pattern.
2. Its main weakness is mixed utterances dominated by one language, which it reads as `same` or `other`. A feature for the share of words from each language could help; it would need a word-language lexicon built inside the training folds.
3. The Spanish determiner asymmetry is a pattern in the children's speech.
4. Some errors are annotation gaps, most visibly in Pérez-Bazán, so the scores there are lower bounds.
