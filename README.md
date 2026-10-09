# OptiQuant

A reproducible pipeline for quantizing open LLMs and benchmarking the trade-offs between
throughput, latency, memory, and accuracy on consumer GPUs.

**Research question:** on a consumer 12 GB GPU, where is the Pareto frontier of quantization
methods for a 7–8B-class LLM?

## Status

Early development. Working today:

- Benchmarking GGUF models with `llama-bench` (prefill and decode throughput, mean ± std over repeats)
- Ingesting results into SQLite with full provenance (llama.cpp build, hardware, KV-cache settings)
- Archiving the raw benchmark JSON so the database can always be rebuilt

Planned: automated quantization sweep (Q8_0 → IQ4_XS), perplexity and `lm-evaluation-harness`
accuracy evaluation, a vLLM runtime comparison (AWQ/GPTQ), and a Pareto-frontier dashboard.

## Requirements

- NVIDIA GPU; developed on an RTX 5070 Ti Laptop GPU (12 GB, Blackwell sm_120)
- CUDA 12.8+ capable driver
- [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA support, `llama-bench` on `PATH`
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

PyTorch is pinned to the CUDA 12.8 wheel index in `pyproject.toml`, since Blackwell GPUs need
cu128+ builds.

## Usage

Benchmark a model, then ingest the results:

```bash
llama-bench -m models/qwen2.5-7b-instruct-q4_k_m.gguf -ngl 99 -p 512 -n 128 -r 5 -o json > bench.json
uv run optiquant-ingest bench.json --model-family Qwen2.5-7B-Instruct --quant Q4_K_M --peak-vram-mib 5200
```

The raw JSON is copied to `results/raw/` and one row per test phase is written to
`data/optiquant.db`. Ingesting the same run twice is rejected.

## Project layout

```
src/optiquant/   schema, DB access, record parsing, ingest CLI
tests/           pytest suite and llama-bench fixture
results/raw/     archived raw benchmark output (source of truth)
data/            SQLite results DB (generated, not committed)
models/          GGUF model files (not committed)
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
