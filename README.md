# ARC TTT Web Demonstration

A small MLOps project for comparing a base language model, in-context learning, and a task-tuned LoRA adapter on ARC-style reasoning tasks.

The goal is not to solve ARC perfectly. The goal is to build a clear workflow for training, registering, evaluating, and testing task-specific adapters with SageMaker.

## Project Idea

For one ARC task, the system compares three modes:

1. **Base model**: the model sees only the test input.
2. **ICL baseline**: the model sees training examples in the prompt.
3. **TTT adapter**: a LoRA adapter is trained for the task, then used for prediction.

Each mode should return:

- predicted output grid
- short reasoning trace
- exact-match score
- cell-level accuracy
- run metadata

## Planned Architecture

```text
ARC task JSON in S3
  -> SageMaker baseline evaluation
  -> SageMaker LoRA adapter training
  -> SageMaker adapter evaluation
  -> result JSON in S3
  -> simple web UI comparison
```

## Core Stack

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter training**: LoRA with torchtune
- **MLOps platform**: Amazon SageMaker
- **Artifact storage**: Amazon S3
- **Model registry**: SageMaker Model Registry
- **Serving candidate**: vLLM
- **Web UI**: simple single-user comparison interface

## First Milestone

Build the smallest end-to-end slice:

1. Create one tiny ARC task file.
2. Upload it to S3.
3. Run a SageMaker job that reads the task.
4. Run base and ICL predictions.
5. Train one LoRA adapter.
6. Evaluate base vs ICL vs adapter.
7. Save one result JSON.

## Current Scripts

Prepare leave-one-out TTT training records:

```bash
python train_ttt_adapter.py \
  --input-folder ../ARC-AGI/data/training \
  --output-adapter-folder outputs/preview \
  --task-id 007bbfb7 \
  --k-train-examples 4 \
  --skip-on-insufficient-demos true \
  --enable-loo true \
  --enable-train-transforms false \
  --seed 42
```

For SageMaker training, use `/opt/ml/input/data/arc` as the input folder and `/opt/ml/model` as the adapter output folder.

## Status

TTT data preprocessing has started. Model training is not implemented yet.
