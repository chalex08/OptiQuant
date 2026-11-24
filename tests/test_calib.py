import gzip
import json

from optiquant.calib import build_calibration, main, read_c4_docs, word_ngrams


def test_ngrams_of_a_simple_sentence():
    assert word_ngrams("the cat sat on the mat", 3) == {
        ("the", "cat", "sat"),
        ("cat", "sat", "on"),
        ("sat", "on", "the"),
        ("on", "the", "mat"),
    }


def test_case_and_punctuation_are_ignored():
    assert (
        word_ngrams("The Cat, sat!", 3) == word_ngrams("the cat sat", 3) == {("the", "cat", "sat")}
    )


def test_wikitext_markers_only_separate_words():
    assert word_ngrams("a well @-@ known band", 3) == {
        ("a", "well", "known"),
        ("well", "known", "band"),
    }


def test_text_shorter_than_n_has_no_ngrams():
    assert word_ngrams("only three words", 4) == set()
    assert word_ngrams("", 1) == set()


def test_default_n_is_13():
    words = " ".join(str(i) for i in range(14))
    assert len(word_ngrams(words)) == 2


def test_overlapping_docs_are_dropped():
    leaked = "word " * 5 + "copied straight out of the wikipedia test split by some web page today"
    clean = "an entirely different document about cooking pasta with garlic and a little olive oil"
    eval_ngrams = word_ngrams(
        "copied straight out of the wikipedia test split by some web page today"
    )

    kept, dropped = build_calibration([leaked, clean], eval_ngrams, target_chars=10**6)
    assert kept == [clean]
    assert dropped == 1


def test_stops_once_target_is_reached_and_is_deterministic():
    docs = [f"document number {i} " * 10 for i in range(50)]
    a, _ = build_calibration(docs, set(), target_chars=500)
    b, _ = build_calibration(docs, set(), target_chars=500)
    assert a == b
    assert sum(map(len, a)) >= 500
    assert sum(map(len, a[:-1])) < 500  # the last doc kept is the one that crossed the target


def test_end_to_end(tmp_path):
    shard = tmp_path / "c4.json.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as f:
        for text in ["first c4 doc", "second c4 doc"]:
            f.write(json.dumps({"text": text, "url": "x"}) + "\n")
    assert list(read_c4_docs(shard)) == ["first c4 doc", "second c4 doc"]

    wiki = tmp_path / "wiki.test.raw"
    wiki.write_text(" = Some Article = \n unrelated text \n", encoding="utf-8")
    out = tmp_path / "calib" / "c4val.txt"

    assert main([str(shard), str(wiki), str(out)]) == 0
    assert sorted(out.read_text(encoding="utf-8").split("\n\n")) == [
        "first c4 doc",
        "second c4 doc",
    ]
