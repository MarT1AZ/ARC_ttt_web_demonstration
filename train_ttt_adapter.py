"""Prepare ARC TTT adapter training data.

This is the first training entrypoint. For now it produces the exact JSONL
records a LoRA training job will consume; model training will be added after
the ARC sampling and prompt format are stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ttt_data import prepare_ttt_records, write_json, write_jsonl


def str_to_bool(value: str) -> bool:
    """Parse true/false CLI flags passed as strings.

    argparse's built-in bool handling is surprising, because bool("false") is
    True. This parser keeps commands like --enable-loo false honest.
    """
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Declare the CLI used locally and inside SageMaker Training Jobs."""
    parser = argparse.ArgumentParser(
        description="Prepare leave-one-out ARC TTT data for LoRA adapter training."
    )

    # Where ARC JSON files are read from. In SageMaker this should usually be
    # /opt/ml/input/data/arc after S3 input data is mounted.
    parser.add_argument("--input-folder", required=True)

    # Where generated training artifacts are written. In SageMaker this should
    # be /opt/ml/model so SageMaker uploads the folder after the job.
    parser.add_argument("--output-adapter-folder", required=True)

    # Use --task-id for one exact ARC task, or omit it and sample --num-tasks.
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--num-tasks", type=int, default=1)

    # Number of task["train"] demonstration pairs to use inside each task.
    parser.add_argument("--k-train-examples", type=int, default=4)

    # If true, tasks with fewer than k demos are skipped and recorded in the
    # manifest. If false, the script proceeds with all available demos.
    parser.add_argument("--skip-on-insufficient-demos", type=str_to_bool, default=True)

    # If true, build leave-one-out ICL-style rows. If false, use direct I/O rows
    # as an ablation.
    parser.add_argument("--enable-loo", type=str_to_bool, default=True)

    # If true, multiply LOO rows with rotate/flip/transpose augmentations.
    parser.add_argument("--enable-train-transforms", type=str_to_bool, default=True)

    # Shuffle demonstration order before selecting k. Keep false for the first
    # clean run; use true later for prompt/order sensitivity experiments.
    parser.add_argument("--shuffle-demos", type=str_to_bool, default=False)

    # Controls task sampling and optional demo shuffling.
    parser.add_argument("--seed", type=int, default=42)

    # Output filenames are configurable so later training jobs can write
    # multiple datasets into the same adapter folder if needed.
    parser.add_argument("--jsonl-name", default="ttt_train.jsonl")
    parser.add_argument("--manifest-name", default="ttt_manifest.json")
    return parser


def main() -> int:
    """Prepare the TTT JSONL and manifest, then print a tiny run summary."""
    args = build_parser().parse_args()

    # In SageMaker Training Jobs, pass /opt/ml/model here so SageMaker packages
    # the generated adapter artifact automatically after training finishes.
    output_dir = Path(args.output_adapter_folder)

    # For now this entrypoint prepares the supervised rows only. The future LoRA
    # trainer should consume this JSONL without changing ARC sampling behavior.
    records, manifest = prepare_ttt_records(
        input_folder=args.input_folder,
        task_id=args.task_id,
        num_tasks=args.num_tasks if args.task_id is None else None,
        k_train_examples=args.k_train_examples,
        skip_on_insufficient_demos=args.skip_on_insufficient_demos,
        enable_loo=args.enable_loo,
        enable_train_transforms=args.enable_train_transforms,
        seed=args.seed,
        shuffle_demos=args.shuffle_demos,
    )

    # The JSONL is the future LoRA training input. The manifest explains exactly
    # how the JSONL was produced.
    write_jsonl(records, output_dir / args.jsonl_name)
    write_json(manifest, output_dir / args.manifest_name)

    # Keep CLI output short because SageMaker logs can get noisy quickly.
    print(f"Wrote {len(records)} training records")
    print(f"JSONL: {output_dir / args.jsonl_name}")
    print(f"Manifest: {output_dir / args.manifest_name}")
    if manifest["skipped_tasks"]:
        print(f"Skipped {len(manifest['skipped_tasks'])} task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
