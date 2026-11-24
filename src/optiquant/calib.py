"""Build the imatrix calibration text from C4, with no n-gram overlap with the eval text.

Usage (from the project root, after downloading the pinned files listed in the README):
    uv run optiquant-calib data/raw/c4/en/c4-validation.00000-of-00008.json.gz \
        data/raw/wikitext-2-raw/wiki.test.raw data/calib/c4val.txt
"""

import argparse
import gzip
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from optiquant.ingest import sha256_file

NGRAM = 13  # long enough that a shared 13-word run means copied text, not common phrasing
SEED = 0
# shortcut: sized in characters (~4 chars/token), not tokens; llama-imatrix --chunks sets the
# exact token budget, so only raise this if imatrix runs out of text.
TARGET_CHARS = 1_200_000


def word_ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    """Return every run of `n` consecutive words in `text`, as a set of tuples.

    Normalize so formatting differences can't hide copied text: lowercase everything, and
    treat a word as a maximal run of ASCII letters and digits (punctuation, whitespace and
    WikiText's " @-@ " / " @,@ " markers all just separate words).

        word_ngrams("The cat, sat on!", 3) == {("the", "cat", "sat"), ("cat", "sat", "on")}

    A text with fewer than `n` words has no n-grams: return an empty set.
    """
    raise NotImplementedError


def read_c4_docs(path: Path) -> Iterator[str]:
    """Yield the "text" field of each line in a gzipped C4 JSON-lines shard."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def build_calibration(
    docs: Iterable[str],
    eval_ngrams: set[tuple[str, ...]],
    target_chars: int = TARGET_CHARS,
    seed: int = SEED,
) -> tuple[list[str], int]:
    """Pick docs in a seeded random order, skipping any that share an n-gram with the eval
    text, until `target_chars` is reached. Returns (kept docs, number dropped for overlap)."""
    pool = list(docs)
    random.Random(seed).shuffle(pool)

    kept, dropped, total = [], 0, 0
    for doc in pool:
        if total >= target_chars:
            break
        if word_ngrams(doc).isdisjoint(eval_ngrams):
            kept.append(doc)
            total += len(doc)
        else:
            dropped += 1
    return kept, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build imatrix calibration text from C4.")
    parser.add_argument("c4_shard", type=Path, help="c4-validation.*.json.gz")
    parser.add_argument("eval_text", type=Path, help="wiki.test.raw (must not leak into calib)")
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    eval_ngrams = word_ngrams(args.eval_text.read_text(encoding="utf-8"))
    kept, dropped = build_calibration(read_c4_docs(args.c4_shard), eval_ngrams)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(kept), encoding="utf-8")
    chars = sum(len(d) for d in kept)
    print(f"Kept {len(kept)} docs ({chars:,} chars), dropped {dropped} for eval overlap")
    print(f"sha256 {sha256_file(args.out)}  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
