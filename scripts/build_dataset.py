"""Build one utterance-level dataset from several CHILDES Spanish-English corpora.

Every corpus is read from its raw CHAT (.cha) files and passed through the SAME
cleaning and labelling rules, so differences between corpora come from the
children and the annotation, not from different preprocessing code.

Output: one row per target-child utterance with
    corpus, child_id, session_id, source_line, session_language,
    age_months, age_wave_months, analysis_age_months, main_sample,
    prev_adult_language, utterance_raw, clean, n_tokens,
    utt_language, label3, label_rule

analysis_age_months: Hoff -> age wave (30/36/42), other corpora -> @ID age.
main_sample: 1 if analysis_age_months <= 42 (up to 3;6), the age range where the
    corpora overlap. Older FerFuLice data are kept (main_sample = 0) for a separate
    descriptive developmental analysis only.
prev_adult_language: language of the most recent non-empty utterance by any
    non-target speaker before this child utterance, in the same file
    (eng / spa / mix, or none if the child speaks first). Derived from the adults'
    own language annotation, not from the child's.

label3 (3-class code-switching label, relative to the context language):
    same  - the utterance is entirely in the session (context) language
    other - the utterance is entirely in the other language (inter-sentential switch)
    mixed - the utterance contains material from both languages (intra-sentential switch)

`utterance_raw` and `label_rule` are kept for transparency/auditing only and must
NOT be used as model input (they contain the annotation codes the labels come from).

Usage:
    python3 scripts/build_dataset.py            # writes data/all_corpora.csv
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "all_corpora.csv"

# ---------------------------------------------------------------- CHAT reading

BULLET = re.compile(r"\x15[^\x15]*\x15")          # media time stamps
PRECODE = re.compile(r"\[- ?([a-z]{3})\]")        # utterance language precode, e.g. [- spa]
AT_S = re.compile(r"\S@s(?::[a-z]{3})?(?![a-z])")  # word from the other language: word@s


def read_chat(path):
    """Yield (line_no, speaker, main_tier_text) for every main tier in a .cha file.

    line_no is the 1-based line on which the main tier starts (continuation lines,
    which begin with a tab, are joined onto it)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    current = None
    for i, line in enumerate(lines, start=1):
        if line.startswith("\t") and current is not None:
            current[2] += " " + line.strip()
            continue
        if current is not None:
            yield tuple(current)
            current = None
        m = re.match(r"\*(\w+):\t(.*)", line)
        if m:
            current = [i, m.group(1), m.group(2)]
    if current is not None:
        yield tuple(current)


def header(path):
    """Return @Languages and a {speaker_code: age_string} map for target children."""
    text = path.read_text(encoding="utf-8")
    langs = re.search(r"^@Languages:\t(.*)$", text, re.M)
    langs = [x.strip() for x in langs.group(1).split(",")] if langs else []
    ages = {}
    for m in re.finditer(r"^@ID:\t([^\n]*)$", text, re.M):
        f = m.group(1).split("|")
        if len(f) > 7 and f[7] == "Target_Child":
            ages[f[2]] = f[3]
    return langs, ages


def age_to_months(age):
    """'2;06.15' -> 30 (years*12 + months; days ignored). '' -> None."""
    m = re.match(r"(\d+);(\d+)?", age or "")
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2) or 0)


# ---------------------------------------------------------------- cleaning

DROP_WORDS = {"xxx", "yyy", "www"}   # unintelligible / untranscribed material
PUNCT_TOKENS = {".", "?", "!", ",", ";", ":", "„", "‡", '"', "“", "”"}


def clean(raw):
    """Turn a CHAT main tier into plain words the child actually said.

    Rules, in order:
      1. remove media bullets (time stamps)
      2. remove every [...] annotation: precodes [- spa], replacements [: x],
         retracing [/] [//] [///], errors [*], overlap [<] [>], comments [^ ..] [= ..]
         (we keep what the child SAID, not the target form in [: x])
      3. remove the < > that mark the scope of those annotations
      4. drop tokens that are not words: &=laugh, &-um, &~frag, &+frag (anything
         starting with &), xxx/yyy/www, omitted words (0word), utterance
         terminators/linkers (+/. +... +< etc.), pauses (.) (..), punctuation
      5. inside a word: strip @-codes (@s, @s:eng, @fp, @i, @l, @x, @c, ...),
         remove ( ) but keep the letters (oop(s) -> oops), turn _ and + into a space
         (french_fries -> french fries), remove prosodic marks (: lengthening, ^, ˈ)
    """
    t = BULLET.sub(" ", raw)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = t.replace("<", " ").replace(">", " ")
    words = []
    for tok in t.split():
        if tok.startswith("&") or tok.startswith("+") or tok.startswith("0"):
            continue
        if re.fullmatch(r"\(\.*\)", tok) or tok in PUNCT_TOKENS:
            continue
        tok = re.sub(r"@[a-z]+(?::[a-z]+)?", "", tok)          # @-codes
        tok = tok.replace("(", "").replace(")", "")
        tok = re.sub(r"[:^ˈˌ≠↑↓]", "", tok)                     # prosody
        tok = tok.strip(".?!,;\"“”„‡")
        tok = tok.replace("_", " ").replace("+", " ")
        for w in tok.split():
            if w and w.lower() not in DROP_WORDS:
                words.append(w)
    return " ".join(words)


# ---------------------------------------------------------------- labelling

def label(raw, session_language):
    """Return (utt_language, label3, label_rule) from the corpus annotation.

    Priority: mixed > other > same.
      mixed : precode [- hun] (Hoff's code for mixed utterances) or any word@s
      other : a language precode different from the session language
      same  : no precode, or a precode equal to the session language
    """
    codes = PRECODE.findall(raw)
    code = codes[0] if codes else None
    if code == "hun":
        return "mix", "mixed", "precode [- hun]"
    if AT_S.search(raw):
        return "mix", "mixed", "word@s"
    if code and code != session_language:
        return code, "other", f"precode [- {code}] != session {session_language}"
    if code:
        return code, "same", f"precode [- {code}] == session"
    return session_language, "same", "no code -> session language"


# ---------------------------------------------------------------- corpora

def hoff():
    """Hoff: biling-{eng,spa}/{2.5,3,3.5}/<child>.cha. Monolingual folder excluded.
    Session language = the language the caregiver was instructed to use (folder).
    Age wave = folder (2.5 -> 30 months)."""
    root = DATA / "Hoff"
    for path in sorted(root.glob("biling-*/*/*.cha")):
        lang = path.parts[-3].split("-")[1]
        wave = round(float(path.parts[-2]) * 12)
        _, ages = header(path)
        yield path, "hoff", f"hoff_{path.stem}", lang, wave, ages, {"CHI"}, \
            str(path.relative_to(root).with_suffix(""))


def ferfulice():
    """FerFuLice: eng/ and spa/ folders = English vs Spanish interlocutor context.
    biling/ (translation and code-switching elicitation tasks) excluded.
    Twins CHI and CHI2; SOL = 'Simon or Leo' (speaker unclear) kept as its own id."""
    root = DATA / "FerFuLice"
    for path in sorted(list(root.glob("eng/*.cha")) + list(root.glob("spa/*.cha"))):
        lang = path.parts[-2]
        _, ages = header(path)
        twin_age = ages.get("CHI") or ages.get("CHI2")   # twins: same age
        if twin_age:
            for code in ("CHI", "CHI2", "SOL"):
                if not ages.get(code):
                    ages[code] = twin_age
        yield path, "ferfulice", None, lang, None, ages, {"CHI", "CHI2", "SOL"}, \
            str(path.relative_to(root).with_suffix(""))


def deuchar():
    """Deuchar: Eng/ (grandmother) and Span/ (father) folders = context language."""
    root = DATA / "Deuchar"
    for folder, lang in (("Eng", "eng"), ("Span", "spa")):
        for path in sorted((root / folder).glob("*.cha")):
            _, ages = header(path)
            yield path, "deuchar", "deuchar_CHI", lang, None, ages, {"CHI"}, \
                str(path.relative_to(root).with_suffix(""))


def perez():
    """Pérez-Bazán: one folder per child. No language instructions were given;
    every file declares Spanish as its main language, so session language = spa."""
    root = DATA / "Perez"
    for path in sorted(root.glob("*/*.cha")):
        langs, ages = header(path)
        lang = langs[0] if langs else "spa"
        yield path, "perez", f"perez_{path.parts[-2]}", lang, None, ages, {"CHI"}, \
            str(path.relative_to(root).with_suffix(""))


CORPORA = [hoff, ferfulice, deuchar, perez]   # Ticio excluded: no switch annotation


MAIN_MAX_AGE = 42   # months; main sample = ages up to 3;6 (where the corpora overlap)


def build():
    fields = ["corpus", "child_id", "session_id", "source_line", "session_language",
              "age_months", "age_wave_months", "analysis_age_months", "main_sample",
              "prev_adult_language", "utterance_raw", "clean", "n_tokens",
              "utt_language", "label3", "label_rule"]
    log = Counter()
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for corpus_fn in CORPORA:
            for path, corpus, child_id, lang, wave, ages, kids, sid in corpus_fn():
                prev_adult = "none"          # reset at the start of every file
                for line_no, spk, raw in read_chat(path):
                    if spk not in kids:
                        # any non-target speaker (parent, investigator, sibling):
                        # remember the language of their last non-empty utterance,
                        # using the same annotation rules as for the child
                        if clean(raw):
                            prev_adult = label(raw, lang)[0]
                        continue
                    log[corpus, "child_utterances"] += 1
                    text = clean(raw)
                    if not text:
                        log[corpus, "dropped_empty_after_cleaning"] += 1
                        continue
                    utt_lang, lab, rule = label(raw, lang)
                    age = age_to_months(ages.get(spk, ""))
                    analysis_age = wave if wave is not None else age  # Hoff: wave
                    main = analysis_age is not None and analysis_age <= MAIN_MAX_AGE
                    w.writerow({
                        "corpus": corpus,
                        "child_id": child_id or f"{corpus}_{spk}",
                        "session_id": f"{corpus}/{sid}",
                        "source_line": line_no,
                        "session_language": lang,
                        "age_months": age if age is not None else "",
                        "age_wave_months": wave if wave is not None else "",
                        "analysis_age_months": analysis_age if analysis_age is not None else "",
                        "main_sample": int(main),
                        "prev_adult_language": prev_adult,
                        "utterance_raw": BULLET.sub("", raw).strip(),
                        "clean": text,
                        "n_tokens": len(text.split()),
                        "utt_language": utt_lang,
                        "label3": lab,
                        "label_rule": rule,
                    })
                    log[corpus, "kept"] += 1
                    log[corpus, lab] += 1
                    log[corpus, "main_sample"] += int(main)
    for k in sorted(log):
        print(k, log[k])
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(build())
