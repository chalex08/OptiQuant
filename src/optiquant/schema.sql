CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id                TEXT PRIMARY KEY,      -- UUID4, generated at ingest
    source_file           TEXT NOT NULL,         -- archived raw llama-bench JSON this row came from

    build_commit          TEXT NOT NULL,         -- llama.cpp build that produced the numbers
    build_number          INTEGER NOT NULL,
    test_time             TEXT NOT NULL,         -- ISO-8601 UTC, as reported by llama-bench

    cpu_model             TEXT NOT NULL,
    gpu_model             TEXT NOT NULL,

    model_family          TEXT NOT NULL,         -- e.g. 'Qwen2.5-7B-Instruct'
    model_size_bytes      INTEGER NOT NULL,
    model_n_params        INTEGER NOT NULL,
    quant_method          TEXT NOT NULL,         -- e.g. 'Q4_K_M', 'Q8_0', 'F16'

    phase                 TEXT NOT NULL CHECK (phase IN ('prefill', 'decode')),
    n_prompt_tokens       INTEGER NOT NULL,
    n_gen_tokens          INTEGER NOT NULL,
    n_gpu_layers          INTEGER NOT NULL,
    n_repeats             INTEGER NOT NULL,

    type_k                TEXT NOT NULL,         -- KV-cache key precision, e.g. 'f16'
    type_v                TEXT NOT NULL,         -- KV-cache value precision
    flash_attn            INTEGER NOT NULL,      -- -1 = auto, 0 = off, 1 = on

    avg_ns                INTEGER NOT NULL,
    stddev_ns             INTEGER NOT NULL,
    avg_tokens_per_sec    REAL NOT NULL,
    stddev_tokens_per_sec REAL NOT NULL,

    peak_vram_bytes       INTEGER,               -- nullable until VRAM capture is automated

    UNIQUE (model_family, quant_method, phase, test_time)
);
