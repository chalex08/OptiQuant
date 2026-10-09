import shutil
from pathlib import Path

from optiquant.db import connect
from optiquant.ingest import main

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench_q4km.json"
ARGS = ["--model-family", "Qwen2.5-7B-Instruct", "--quant", "Q4_K_M"]


def test_ingest_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results/raw/ and the DB land in a throwaway dir
    src = tmp_path / "bench.json"
    shutil.copyfile(FIXTURE, src)
    db = tmp_path / "test.db"

    assert main([str(src), *ARGS, "--peak-vram-mib", "5200", "--db", str(db)]) == 0

    archived = tmp_path / "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"
    assert archived.read_bytes() == FIXTURE.read_bytes()

    conn = connect(db)
    rows = conn.execute("SELECT source_file, peak_vram_bytes FROM benchmark_runs").fetchall()
    conn.close()
    assert len(rows) == 2
    assert all(
        r["source_file"] == "results/raw/20260824T123222Z_Qwen2.5-7B-Instruct_Q4_K_M.json"
        for r in rows
    )
    assert all(r["peak_vram_bytes"] == 5200 * 1024**2 for r in rows)


def test_reingesting_inserts_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "bench.json"
    shutil.copyfile(FIXTURE, src)
    db = tmp_path / "test.db"

    assert main([str(src), *ARGS, "--db", str(db)]) == 0
    assert main([str(src), *ARGS, "--db", str(db)]) == 1

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 2
    conn.close()
