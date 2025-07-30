#!/usr/bin/env python3
#uses spacey and tf-idf to pull out the most important keywords from the transcripts
#displays total word count across all transcripts 
#original keyword pulling script


"""
run_keywords.py — TF-IDF keyword miner (uni- to quad-grams)
────────────────────────────────────────────────────────────
• Streams every file in ./transcripts
• Removes greetings, single-letter tokens, and numbers
• Keeps n-grams (1-4) that include ≥ 1 NOUN / PROPN / ADJ
• Ranks phrases by summed TF-IDF across the corpus
• Prints one clean 4-column table

SET-UP
$ pip install -U spacy scikit-learn numpy prettytable
$ python -m spacy download en_core_web_sm
$ python run_keywords.py
"""
from __future__ import annotations

import pathlib
import re
import sys
import warnings
from collections import Counter
from datetime import datetime
from typing import Iterable, List

import numpy as np
import sklearn
import spacy
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────── CONFIG ────────────────────
TRANSCRIPTS_DIR = pathlib.Path("transcripts")
MAX_CHUNK_LINES = 4000
TOPN            = {1: 500, 2: 500, 3: 500, 4: 500}
# ────────────────────────────────────────────────

# Silence the sklearn stop-word tokenisation warning
warnings.filterwarnings(
    "ignore",
    message="Your stop_words may be inconsistent",
    category=UserWarning,
    module=sklearn.__name__,
)

# ------------ 1  NLP initialisation ------------
NLP = spacy.load("en_core_web_sm", disable=["parser", "textcat"])
SPACY_STOP = NLP.Defaults.stop_words

FILLER_WORDS = {
    "yeah", "yah", "yea", "ok", "okay", "alright", "uh", "um", "hmm",
    "right", "like", "literally", "basically", "actually", "gonna",
    "going", "gotta", "kinda", "sorta",
}

GREETINGS = {
    "hi", "hello", "hey", "bye", "good", "morning", "afternoon",
    "evening", "goodmorning", "goodafternoon", "goodevening",
    "goodbye", "thanks", "thank",
}

CONTRACTION_REPL = {
    r"\bdon['']t\b": "dont",
    r"\bdoesn['']t\b": "doesnt",
    r"\bdidn['']t\b": "didnt",
    r"\bwon['']t\b": "wont",
    r"\bcan['']t\b": "cant",
    r"\baren['']t\b": "arent",
    r"\blet['']s\b": "lets",
    r"\bI['']m\b": "im",
    r"\bI['']ve\b": "ive",
    r"\bI['']ll\b": "ill",
}
CONTRACTION_TOKENS = {
    "dont", "doesnt", "didnt", "wont", "cant", "arent",
    "lets", "im", "ive", "ill",
}

CUSTOM_STOP = SPACY_STOP | FILLER_WORDS | GREETINGS | CONTRACTION_TOKENS | {"s", "pm", "am"}

BAD_ENTS = {
    "PERSON", "ORG", "NORP", "GPE", "LOC", "FAC",
    "PRODUCT", "EVENT", "LANGUAGE", "DATE", "TIME",
}

# regex helpers
DIGIT_RE = re.compile(r"^\d+$")          # pure numbers
ALPHA1_RE = re.compile(r"^[A-Za-z]$")    # single letters

# ------------ 2  Pre-processing helpers ---------
def normalise_contractions(text: str) -> str:
    for pat, repl in CONTRACTION_REPL.items():
        text = re.sub(pat, repl, text, flags=re.I)
    return text


SPEAKER_RE = re.compile(
    r"^\s*[-–—]?\s*[\w\s]{1,40}\[?[^\n:]{0,20}\]?\s*:",
    flags=re.M,
)

def strip_header(raw: str) -> str:
    body = re.split(r"Transcript Content\s*:?", raw, flags=re.I)[-1]
    body = SPEAKER_RE.sub("", body)
    return normalise_contractions(body)


def load_files() -> Iterable[tuple[str, Iterable[str]]]:
    for fp in sorted(TRANSCRIPTS_DIR.glob("*")):
        if fp.is_file():
            yield fp.name, fp.open(encoding="utf-8", errors="ignore")

# ------------ 3  Filtering utilities ------------
def reject_token(tok: spacy.tokens.Token) -> bool:
    return (
        tok.lemma_.lower() in CUSTOM_STOP
        or tok.is_punct
        or tok.is_space
        or tok.ent_type_ in BAD_ENTS
        or tok.pos_ == "NUM"
        or tok.like_num
        or DIGIT_RE.fullmatch(tok.text)
        or ALPHA1_RE.fullmatch(tok.text)
    )


def accept_ngram(tokens: List[spacy.tokens.Token]) -> bool:
    if GREETINGS & {t.lemma_.lower() for t in tokens}:
        return False
    if reject_token(tokens[0]) or reject_token(tokens[-1]):
        return False
    if not any(t.pos_ in {"NOUN", "PROPN", "ADJ"} for t in tokens):
        return False
    return not any(reject_token(t) for t in tokens)

# ------------ 4  Counting n-grams ---------------
def update_counters(doc: spacy.tokens.Doc, ctrs: List[Counter]) -> None:
    L = len(doc)
    for n in range(1, 5):
        for i in range(L - n + 1):
            span = doc[i : i + n]
            if not accept_ngram(list(span)):
                continue
            phrase = " ".join(t.lemma_.lower() for t in span)
            ctrs[n - 1][phrase] += 1


def count_all() -> List[Counter]:
    master = [Counter() for _ in range(4)]
    for _fname, fh in load_files():
        buf: List[str] = []
        for line in fh:
            buf.append(line)
            if len(buf) >= MAX_CHUNK_LINES:
                update_counters(NLP(strip_header(" ".join(buf))), master)
                buf.clear()
        if buf:
            update_counters(NLP(strip_header(" ".join(buf))), master)
    return master

# ------------ 5  TF-IDF ranking -----------------
def corpus_docs() -> List[str]:
    docs: List[str] = []
    for p in TRANSCRIPTS_DIR.glob("*"):
        if p.is_file():
            try:
                docs.append(strip_header(p.read_text(encoding="utf-8", errors="ignore")))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  {p.name}: {exc}", file=sys.stderr)
    return docs


def top_tfidf_phrases(docs: List[str], n: int, k: int) -> List[str]:
    """
    Rank n-grams by summed TF-IDF with a strict token pattern:
    • only alphabetic tokens, ≥ 2 chars ⇒ no numbers / single letters
    """
    vect = TfidfVectorizer(
        ngram_range=(n, n),
        min_df=2,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",   # crucial line
        stop_words=list(CUSTOM_STOP),
    )
    X = vect.fit_transform(docs)
    if X.shape[1] == 0:
        return []
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms  = np.array(vect.get_feature_names_out())
    idx    = scores.argsort()[::-1][:k]
    return terms[idx].tolist()

# ------------ 6  Table helpers ------------------
def pad_rows(cols: dict[int, List[str]]) -> List[List[str]]:
    m = max(len(v) for v in cols.values())
    return [
        [
            cols[1][i] if i < len(cols[1]) else "",
            cols[2][i] if i < len(cols[2]) else "",
            cols[3][i] if i < len(cols[3]) else "",
            cols[4][i] if i < len(cols[4]) else "",
        ]
        for i in range(m)
    ]

# ------------ 7  MAIN ---------------------------
def main() -> None:
    counters = count_all()   # Get raw counts for each n-gram

    docs = corpus_docs()
    if not docs:
        print("❌  No transcripts found.", file=sys.stderr)
        sys.exit(1)

    # Compute total word count across all transcripts
    total_word_count = sum(len(doc.split()) for doc in docs)

    # Get top phrases and their counts for each n-gram
    cols = {}
    counts = {}
    for n in range(1, 5):
        phrases = top_tfidf_phrases(docs, n, TOPN[n])
        cols[n] = phrases
        # Get counts for each phrase, incrementing by 1 so minimum is 1
        counts[n] = [counters[n-1][phrase] + 1 for phrase in phrases]

    # Prepare PrettyTable with count columns
    table = PrettyTable(
        ["UNIGRAMS", "UNI-COUNT", "BIGRAMS", "BI-COUNT", "TRIGRAMS", "TRI-COUNT", "QUADGRAMS", "QUAD-COUNT"], hrules=True
    )
    table.max_table_width = 120
    for f in table.field_names:
        table.align[f] = "l"
    # Pad rows for phrases and counts
    m = max(len(cols[n]) for n in range(1, 5))
    for i in range(m):
        row = [
            cols[1][i] if i < len(cols[1]) else "",
            counts[1][i] if i < len(counts[1]) else "",
            cols[2][i] if i < len(cols[2]) else "",
            counts[2][i] if i < len(counts[2]) else "",
            cols[3][i] if i < len(cols[3]) else "",
            counts[3][i] if i < len(counts[3]) else "",
            cols[4][i] if i < len(cols[4]) else "",
            counts[4][i] if i < len(counts[4]) else "",
        ]
        table.add_row(row)

    banner = "=" * (len(table.get_string().splitlines()[0]) or 80)
    print("\n" + banner)
    print("KEYWORD UNIVERSE –", datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(banner)
    print(table)
    print(banner + "\n")
    print(f"Total word count across all transcripts: {total_word_count}")

    # Export the table to a CSV file
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for CSV export. Please install it with 'pip install pandas'.")
        return

    # Pad all columns to the same length for DataFrame creation
    max_len = max(len(cols[n]) for n in range(1, 5))
    def pad(lst, fill):
        return lst + [fill] * (max_len - len(lst))
    df = pd.DataFrame({
        "UNIGRAMS": pad(cols[1], ""),
        "UNI-COUNT": pad(counts[1], 1),
        "BIGRAMS": pad(cols[2], ""),
        "BI-COUNT": pad(counts[2], 1),
        "TRIGRAMS": pad(cols[3], ""),
        "TRI-COUNT": pad(counts[3], 1),
        "QUADGRAMS": pad(cols[4], ""),
        "QUAD-COUNT": pad(counts[4], 1),
    })
    output_filename_csv = "keyword_universe.csv"
    output_filename_xlsx = "keyword_universe.xlsx"
    df.to_csv(output_filename_csv, index=False)
    df.to_excel(output_filename_xlsx, index=False)
    print(f"Table exported to '{output_filename_csv}' and '{output_filename_xlsx}'.")
    print(f"Total word count across all transcripts: {total_word_count}")


if __name__ == "__main__":
    if not TRANSCRIPTS_DIR.exists():
        print("❌  'transcripts' directory not found.", file=sys.stderr)
        sys.exit(1)
    main()
