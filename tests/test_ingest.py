import hashlib
import json
import shutil
from pathlib import Path

import pytest

from optiquant.db import connect
from optiquant.ingest import main, sha256_file

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench_q4km.json"
GGUF_BYTES = b"stand-in for a GGUF file"
ARGS = [
    "--model-family", "Qwen2.5-7B-Instruct",
    "--quant", "Q4_K_M",
    "--recipe", "third-party",
    "--gguf", "qwen2.5-7b-instruct-q4_k_m.gguf",  # same file name the fixture benchmarked
]  # fmt: skip
ARCHIVED = "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results/raw/, the DB and the GGUF live in a throwaway dir
    (tmp_path / "qwen2.5-7b-instruct-q4_k_m.gguf").write_bytes(GGUF_BYTES)


def test_ingest_end_to_end(tmp_path):
    src = tmp_path / "bench.json"
    shutil.copyfile(FIXTURE, src)
    db = tmp_path / "test.db"

    assert main([str(src), *ARGS, "--peak-vram-mib", "5200", "--db", str(db)]) == 0

    archived = tmp_path / "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"
    assert archived.read_bytes() == FIXTURE.read_bytes()

    conn = connect(db)
    rows = conn.execute(
        "SELECT source_file, peak_vram_bytes, recipe, gguf_sha256 FROM benchmark_runs"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert all(
        r["source_file"] == "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"
        for r in rows
    )
    assert all(r["peak_vram_bytes"] == 5200 * 1024**2 for r in rows)
    assert all(r["recipe"] == "third-party" for r in rows)
    assert all(r["gguf_sha256"] == hashlib.sha256(GGUF_BYTES).hexdigest() for r in rows)

    meta = json.loads((tmp_path / ARCHIVED).with_suffix(".meta.json").read_text("utf-8"))
    assert meta == {
        "model_family": "Qwen2.5-7B-Instruct",
        "quant": "Q4_K_M",
        "recipe": "third-party",
        "gguf_file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "gguf_sha256": hashlib.sha256(GGUF_BYTES).hexdigest(),
        "peak_vram_mib": 5200.0,
    }


def test_utf16_input_is_archived_as_utf8(tmp_path):
    src = tmp_path / "bench.json"
    src.write_bytes(FIXTURE.read_text(encoding="utf-8").encode("utf-16"))  # BOM + UTF-16LE
    db = tmp_path / "test.db"

    assert main([str(src), *ARGS, "--db", str(db)]) == 0

    archived = tmp_path / "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"
    assert archived.read_bytes() == FIXTURE.read_bytes()

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 2
    conn.close()


def test_reingesting_inserts_nothing(tmp_path):
    src = tmp_path / "bench.json"
    shutil.copyfile(FIXTURE, src)
    db = tmp_path / "test.db"

    assert main([str(src), *ARGS, "--db", str(db)]) == 0
    assert main([str(src), *ARGS, "--db", str(db)]) == 1

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 2
    conn.close()


def test_gguf_that_was_not_benchmarked_is_rejected(tmp_path):
    src = tmp_path / "bench.json"
    shutil.copyfile(FIXTURE, src)
    (tmp_path / "other.gguf").write_bytes(GGUF_BYTES)
    args = [*ARGS[:-1], "other.gguf"]

    assert main([str(src), *args, "--db", str(tmp_path / "test.db")]) == 1
    assert not (tmp_path / "results").exists()  # rejected before anything was archived


def test_sha256_of_empty_file(tmp_path):
    f = tmp_path / "empty"
    f.write_bytes(b"")
    assert sha256_file(f) == hashlib.sha256(b"").hexdigest()


def test_sha256_spanning_several_chunks(tmp_path):
    data = bytes(range(256)) * 41  # 10,496 bytes: not a multiple of the chunk size
    f = tmp_path / "data.bin"
    f.write_bytes(data)
    assert sha256_file(f, chunk_size=1000) == hashlib.sha256(data).hexdigest()
