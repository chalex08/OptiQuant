"""Ingest llama-bench JSON output into the results DB.

Usage (from the project root):
    llama-bench -m models/<file>.gguf -ngl 99 -p 512 -n 128 -r 5 -o json > bench.json
    uv run optiquant-ingest bench.json --model-family Qwen2.5-7B-Instruct --quant Q4_K_M
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from optiquant.db import DEFAULT_DB_PATH, connect
from optiquant.records import insert_run, parse_record

RAW_DIR = Path("results/raw")


def read_bench_text(json_path: Path) -> str:
    """Read llama-bench output whatever encoding the shell wrote it in.

    Windows PowerShell 5.1's `>` writes UTF-16 with a BOM; utf-8-sig handles UTF-8 with or
    without a BOM.
    """
    raw = json_path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def archive_raw(text: str, records: list[dict], model_family: str, quant: str) -> Path:
    """Write the raw JSON into results/raw/ as UTF-8 under a deterministic name.

    The content is kept exactly as llama-bench produced it; only the encoding is normalized,
    so archives are plain text in git regardless of which shell redirected the output.
    The name comes from the run's own timestamp (not "now"), so re-archiving the same run
    maps to the same file instead of creating a duplicate.
    """
    stamp = records[0]["test_time"].replace("-", "").replace(":", "")
    safe_family = model_family.replace("/", "--")
    dest = RAW_DIR / f"{stamp}_{safe_family}_{quant}.json"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(text.encode("utf-8"))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest llama-bench JSON into the results DB.")
    parser.add_argument("json_path", type=Path, help="file produced by `llama-bench -o json`")
    parser.add_argument("--model-family", required=True, help="e.g. Qwen2.5-7B-Instruct")
    parser.add_argument("--quant", required=True, help="e.g. Q4_K_M")
    parser.add_argument(
        "--peak-vram-mib",
        type=float,
        default=None,
        help="peak VRAM observed in nvidia-smi, in MiB (optional)",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    text = read_bench_text(args.json_path)
    records = json.loads(text)
    if not records:
        print("No benchmark records in file.", file=sys.stderr)
        return 1

    peak_vram_bytes = int(args.peak_vram_mib * 1024**2) if args.peak_vram_mib is not None else None
    archived = archive_raw(text, records, args.model_family, args.quant)

    conn = connect(args.db)
    try:
        with conn:  # one transaction: every record in the file goes in, or none do
            for record in records:
                row = parse_record(
                    record,
                    model_family=args.model_family,
                    quant_method=args.quant,
                    source_file=archived.as_posix(),
                    peak_vram_bytes=peak_vram_bytes,
                )
                insert_run(conn, row)
    except sqlite3.IntegrityError as e:
        print(f"Rejected, nothing inserted (already ingested?): {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(f"Inserted {len(records)} row(s) from {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
