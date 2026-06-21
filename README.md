# MORQA-RCU

Medical Open-Response Question Answering.

## Environment (Pixi)

This project now uses [Pixi](https://pixi.sh) for dependency and task management.

### Install dependencies

```bash
pixi install
```

### Run tests

```bash
pixi run test
pixi run test-verbose
```

### Run analysis/evaluation tasks

```bash
pixi run analyze
pixi run eval-iaa-classification
pixi run similarity-biobert
pixi run similarity-tfidf
```

## Python version

The Pixi workspace targets Python `>=3.10,<3.13` for compatibility with current ML dependencies.
