# MORQA-RCU

Response Content Units: Medical Open-Response Question Answering

## Environment (Pixi)

This project uses [Pixi](https://pixi.sh) for dependency and task management.

## Prerequisites

Before running extraction/evaluation tasks, make sure both are set up:

1. Dataset files under `data/rcu-en/` (see [Dataset](#dataset))
2. API keys for LLM tasks (see [Environment variables](#environment-variables))

### Install dependencies

```bash
pixi install
```

### Run extraction tasks

```bash
pixi run extract-gpt
pixi run extract-gemini
pixi run extract-qwen
pixi run extract-deepseek
pixi run extract-azure-deepseek
pixi run extract-rule
pixi run mrc-train
pixi run extract-mrc  # Requires a trained model
pixi run extract-biobert  # Similarity-based extraction using BioBERT embeddings
```

### Run extraction via CLI directly

```bash
pixi run python -m src.extraction.cli --help
pixi run python -m src.extraction.cli gpt --model-name gpt-4o
pixi run python -m src.extraction.cli mrc-train --max-steps 5
```

### Run analysis/evaluation tasks

```bash
pixi run eval-iaa-classification
pixi run eval-question-classification
pixi run eval-binary-answer-classification
pixi run eval-question-extraction
pixi run eval-answer-extraction
pixi run eval-iaa-extraction
pixi run eval-prognosis-extraction
pixi run similarity-biobert
pixi run similarity-tfidf
```

### Run evaluation via CLI directly

```bash
pixi run python -m src.eval.cli --help
pixi run python -m src.eval.cli iaa-classification
pixi run python -m src.eval.cli run answer-extraction
```

### Run tests

```bash
pixi run test
pixi run test-verbose
```

## Dataset

There is currently no automated dataset download script in this repository.

Dataset source:

- https://osf.io/kcv2n/files/osfstorage (under `rcu-annotations-gem2026` directory)

Expected location:

- `data/rcu-en/`

Expected file naming pattern:

- `<dataset>_<split>.json` (for example: `iiyi_test_gold.json`, `woundcare_valid_systems.json`)

If these files are missing, extraction/evaluation commands will fail or skip missing prediction files.

## Environment variables

LLM extraction tasks require API keys.

Required keys:

- `AZURE_API_KEY` (GPT, Azure DeepSeek)
- `QWEN_API_KEY`
- `DEEPSEEK_API_KEY`
- `GOOGLE_API_KEY`

You can provide keys in either of these ways:

1. `.env` file at project root (recommended)
2. Exported shell environment variables

### Option 1: `.env` file (recommended)

Extraction/eval CLIs auto-load `.env` from the project root.

1. Copy `.env.example` to `.env`
2. Fill in your keys

```bash
cp .env.example .env
```

### Option 2: export in shell

```bash
export AZURE_API_KEY=...
export QWEN_API_KEY=...
export DEEPSEEK_API_KEY=...
export GOOGLE_API_KEY=...
```

Then run tasks normally with Pixi (for example, `pixi run extract-gpt`).

Note: `pixi run --clean-env ...` will not inherit shell variables unless you explicitly provide them.

## Python version

The Pixi workspace targets Python `>=3.10,<3.13` for compatibility with current ML dependencies.
