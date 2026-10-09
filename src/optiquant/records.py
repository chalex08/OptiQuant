import sqlite3
import uuid


def parse_record(
    record: dict,
    *,
    model_family: str,
    quant_method: str,
    source_file: str,
    peak_vram_bytes: int | None = None,
) -> dict:
    if record["n_prompt"] == 0 and record["n_gen"] == 0:
        raise ValueError("Record must have either prefill or decode tokens")
    elif record["n_prompt"] > 0 and record["n_gen"] > 0:
        raise ValueError("Prefill and decode phases must be separate records")
    elif len(record["samples_ns"]) != len(record["samples_ts"]):
        raise ValueError("Mismatched sample counts in record")

    return {
        "run_id": str(uuid.uuid4()),
        "source_file": source_file,
        "build_commit": record["build_commit"],
        "build_number": record["build_number"],
        "test_time": record["test_time"],
        "cpu_model": record["cpu_info"],
        "gpu_model": record["gpu_info"],
        "model_family": model_family,
        "model_size_bytes": record["model_size"],
        "model_n_params": record["model_n_params"],
        "quant_method": quant_method,
        "phase": "prefill" if record["n_prompt"] > 0 else "decode",
        "n_prompt_tokens": record["n_prompt"],
        "n_gen_tokens": record["n_gen"],
        "n_gpu_layers": record["n_gpu_layers"],
        "n_repeats": len(record["samples_ns"]),
        "type_k": record["type_k"],
        "type_v": record["type_v"],
        "flash_attn": record["flash_attn"],
        "avg_ns": record["avg_ns"],
        "stddev_ns": record["stddev_ns"],
        "avg_tokens_per_sec": record["avg_ts"],
        "stddev_tokens_per_sec": record["stddev_ts"],
        "peak_vram_bytes": peak_vram_bytes,
    }


def insert_run(conn: sqlite3.Connection, row: dict) -> None:
    query = (
        f"INSERT INTO benchmark_runs ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})"
    )
    conn.execute(query, row)
