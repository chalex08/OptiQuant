import copy
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from optiquant.db import connect, table_columns
from optiquant.records import insert_run, parse_record

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench_q4km.json"
META = {
    "model_family": "Qwen2.5-7B-Instruct",
    "quant_method": "Q4_K_M",
    "source_file": "results/raw/x.json",
}


@pytest.fixture
def records() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


def test_keys_match_table_columns_exactly(records, conn):
    row = parse_record(records[0], **META)
    assert set(row) == set(table_columns(conn))


def test_prefill_record(records):
    row = parse_record(records[0], **META, peak_vram_bytes=5_000_000_000)
    assert row["phase"] == "prefill"
    assert row["n_prompt_tokens"] == 512
    assert row["n_gen_tokens"] == 0
    assert row["avg_tokens_per_sec"] == pytest.approx(3236.612381)
    assert row["stddev_tokens_per_sec"] == pytest.approx(545.767954)
    assert row["avg_ns"] == 162835320
    assert row["n_repeats"] == 5
    assert row["peak_vram_bytes"] == 5_000_000_000


def test_decode_record(records):
    row = parse_record(records[1], **META)
    assert row["phase"] == "decode"
    assert row["n_prompt_tokens"] == 0
    assert row["n_gen_tokens"] == 128
    assert row["avg_tokens_per_sec"] == pytest.approx(77.843368)
    assert row["peak_vram_bytes"] is None


def test_provenance_and_identity_fields(records):
    row = parse_record(records[0], **META)
    assert row["build_commit"] == "6b5c2efb4"
    assert row["build_number"] == 10306
    assert row["test_time"] == "2026-08-24T12:32:22Z"
    assert row["cpu_model"] == "Intel(R) Core(TM) Ultra 9 275HX"
    assert row["gpu_model"] == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    assert row["model_family"] == "Qwen2.5-7B-Instruct"
    assert row["quant_method"] == "Q4_K_M"
    assert row["source_file"] == "results/raw/x.json"
    assert row["model_size_bytes"] == 4677120000
    assert row["model_n_params"] == 7615616512
    assert row["n_gpu_layers"] == 99
    assert row["type_k"] == "f16" and row["type_v"] == "f16"
    assert row["flash_attn"] == -1


def test_run_id_is_a_fresh_uuid(records):
    a = parse_record(records[0], **META)["run_id"]
    b = parse_record(records[0], **META)["run_id"]
    assert isinstance(a, str)
    uuid.UUID(a)  # raises if not a valid UUID
    assert a != b


def test_combined_prompt_and_gen_test_is_rejected(records):
    mixed = copy.deepcopy(records[0])
    mixed["n_gen"] = 128
    with pytest.raises(ValueError):
        parse_record(mixed, **META)


def test_record_with_neither_prompt_nor_gen_is_rejected(records):
    empty = copy.deepcopy(records[0])
    empty["n_prompt"] = 0
    with pytest.raises(ValueError):
        parse_record(empty, **META)


def test_insert_and_read_back(records, conn):
    for r in records:
        insert_run(conn, parse_record(r, **META))
    rows = conn.execute(
        "SELECT phase, avg_tokens_per_sec FROM benchmark_runs ORDER BY phase"
    ).fetchall()
    assert [r["phase"] for r in rows] == ["decode", "prefill"]
    assert rows[0]["avg_tokens_per_sec"] == pytest.approx(77.843368)


def test_same_run_twice_violates_unique_constraint(records, conn):
    insert_run(conn, parse_record(records[0], **META))
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(conn, parse_record(records[0], **META))
