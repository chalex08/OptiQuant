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
uv run optiquant-ingest bench.json --model-family Qwen2.5-7B-Instruct --quant Q4_K_M \
    --recipe third-party --gguf models/qwen2.5-7b-instruct-q4_k_m.gguf --peak-vram-mib 5200
```

`--recipe` labels where the GGUF came from (`third-party`, `unquantized`, `no-imatrix`, or
`imatrix-<calibration-name>`); `--gguf` is hashed (SHA-256) so every row identifies the exact
file it measured. The raw JSON is copied to `results/raw/` with a `.meta.json` sidecar holding
everything llama-bench doesn't record, and one row per test phase is written to
`data/optiquant.db`. Ingesting the same run twice is rejected.

## Calibration and evaluation data

The importance matrix (imatrix) is calibrated on C4 English validation text; accuracy is
measured on WikiText-2 test. Both are pinned to exact upstream revisions:

- `en/c4-validation.00000-of-00008.json.gz` from `allenai/c4` @ `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`,
  SHA-256 `1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f`
- `wikitext-2-raw-v1.zip` from `ggml-org/ci` @ `927b3642933080f1b0e811e2f916e14c292992f9`,
  SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`

```bash
hf download allenai/c4 en/c4-validation.00000-of-00008.json.gz --repo-type dataset \
    --revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 --local-dir data/raw/c4
hf download ggml-org/ci wikitext-2-raw-v1.zip --repo-type dataset \
    --revision 927b3642933080f1b0e811e2f916e14c292992f9 --local-dir data/raw
unzip data/raw/wikitext-2-raw-v1.zip -d data/raw
uv run optiquant-calib data/raw/c4/en/c4-validation.00000-of-00008.json.gz \
    data/raw/wikitext-2-raw/wiki.test.raw data/calib/c4val.txt
```

`optiquant-calib` takes a seeded random sample of C4 documents (~1.2M characters) and drops
any document sharing a 13-word sequence with the WikiText-2 test set, so the calibration text
can't leak the evaluation text into the quantization.

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
