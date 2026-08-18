"""ARC task preprocessing for task-specific TTT adapter training.

This module builds the tiny training set used for one ARC task:
selected demo pairs -> leave-one-out ICL prompts -> optional transforms.
It intentionally has no model or SageMaker dependency.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


Grid = list[list[int]]
Pair = dict[str, Grid]
Transform = Callable[[Grid], Grid]

# These are the first geometric transforms we use for task-time training data.
# Each transform is invertible, which matters later when inference voting maps
# transformed predictions back to the original grid orientation.
DEFAULT_TRANSFORMS = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_h",
    "flip_v",
    "transpose",
)


def load_task(task_path: str | Path) -> dict[str, Any]:
    """Load one ARC JSON task from disk.

    ARC task files have two top-level lists:
    - train: known input/output demonstrations for the puzzle
    - test: held-out query examples used for scoring
    """
    with Path(task_path).open("r", encoding="utf-8") as f:
        task = json.load(f)

    # Fail early if the file is not in the expected ARC format.
    if "train" not in task or "test" not in task:
        raise ValueError(f"ARC task is missing train/test keys: {task_path}")
    return task


def list_task_paths(input_folder: str | Path) -> list[Path]:
    """Return all ARC task JSON files from one split folder."""
    return sorted(Path(input_folder).glob("*.json"))


def select_task_paths(
    input_folder: str | Path,
    task_id: str | None = None,
    num_tasks: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Choose task files either by exact task ID or by seeded sampling."""
    paths = list_task_paths(input_folder)

    # Exact task selection is useful while debugging one known puzzle.
    if task_id:
        # ARC task IDs are the JSON filename without ".json".
        target = Path(input_folder) / f"{task_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"Task not found: {target}")
        return [target]

    if num_tasks is None:
        return paths
    if num_tasks < 1:
        raise ValueError("--num-tasks must be greater than 0")

    # Seeded sampling gives reproducible experiments when we run N tasks from
    # the 400-task training/evaluation split.
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    # Return sorted paths so output order is stable after sampling.
    return sorted(shuffled[: min(num_tasks, len(shuffled))])


def task_id_from_path(task_path: str | Path) -> str:
    """Convert data/training/007bbfb7.json into task ID 007bbfb7."""
    return Path(task_path).stem


def select_demo_pairs(
    task: dict[str, Any],
    k_train_examples: int,
    seed: int = 42,
    shuffle_demos: bool = False,
    skip_on_insufficient_demos: bool = True,
) -> tuple[list[Pair], list[int]] | None:
    """Select k demonstration pairs from task["train"].

    Returns both the selected pairs and their original indices. The indices are
    stored in the manifest so we can later reproduce exactly which demos were
    used for an adapter.
    """
    if k_train_examples < 1:
        raise ValueError("--k-train-examples must be greater than 0")

    demos = task["train"]

    # If skip is enabled, this task should not be used for this run because it
    # cannot provide the requested number of demonstrations.
    if len(demos) < k_train_examples and skip_on_insufficient_demos:
        # Keeps k fixed across an experiment instead of silently mixing k=4,
        # k=3, etc. The caller records the skipped task in the manifest.
        return None

    indices = list(range(len(demos)))
    if shuffle_demos:
        # Seeded shuffle lets us test prompt/order sensitivity reproducibly.
        random.Random(seed).shuffle(indices)

    # If skip is disabled and k is too large, we use every available demo.
    selected_indices = indices[: min(k_train_examples, len(indices))]
    selected = [deepcopy(demos[i]) for i in selected_indices]
    return selected, selected_indices


def grid_to_text(grid: Grid) -> str:
    """Serialize an ARC grid compactly so prompts stay short."""
    return json.dumps(grid, separators=(",", ":"))


def build_icl_prompt(demo_pairs: list[Pair], query_grid: Grid) -> str:
    """Build the text prompt for both ICL inference and LOO TTT training."""
    # This same prompt shape is used for ICL baselines and LOO TTT rows:
    # examples first, then one query grid whose output is the training target.
    lines = [
        "You are solving an ARC grid transformation task.",
        "Infer the rule from the examples, then answer the query.",
        "Return only JSON in this exact format: {\"output\": [[...]]}",
        "",
    ]

    # Demonstration examples show the model the task-specific transformation.
    for idx, pair in enumerate(demo_pairs, start=1):
        lines.extend(
            [
                f"Example {idx} input:",
                grid_to_text(pair["input"]),
                f"Example {idx} output:",
                grid_to_text(pair["output"]),
                "",
            ]
        )

    # The query is the grid whose output should be generated. During LOO
    # training the target is known; during real inference the target is hidden.
    lines.extend(["Query input:", grid_to_text(query_grid), "Answer:"])
    return "\n".join(lines)


def build_target(output_grid: Grid) -> str:
    """Build the supervised target text the model should learn to emit."""
    return json.dumps({"output": output_grid}, separators=(",", ":"))


def identity(grid: Grid) -> Grid:
    """No-op transform used so non-augmented and augmented code share a path."""
    return deepcopy(grid)


def rotate90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise."""
    return [list(row) for row in zip(*grid)][::-1]


def flip_h(grid: Grid) -> Grid:
    """Flip grid horizontally, left-to-right."""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Flip grid vertically, top-to-bottom."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Reflect grid across its main diagonal."""
    return [list(row) for row in zip(*grid)]


# Transform names are stored in JSONL records so training data can be audited.
TRANSFORMS: dict[str, Transform] = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
}

# Inference/evaluation needs this to normalize transformed predictions back to
# the original task orientation before scoring or voting.
INVERSE_TRANSFORMS = {
    "identity": "identity",
    "rotate90": "rotate270",
    "rotate180": "rotate180",
    "rotate270": "rotate90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
}


def transform_pair(pair: Pair, transform_name: str) -> Pair:
    """Apply one transform to both the input and output grid of a pair."""
    fn = TRANSFORMS[transform_name]
    return {"input": fn(pair["input"]), "output": fn(pair["output"])}


def transform_pairs(pairs: list[Pair], transform_name: str) -> list[Pair]:
    """Apply one transform to a list of demonstration pairs."""
    return [transform_pair(pair, transform_name) for pair in pairs]


def demo_order_variants(
    demo_pairs: list[Pair],
    demo_indices: list[int],
    num_demo_permutations: int,
    seed: int,
    task_id: str,
    heldout_index: int,
    transform_name: str,
) -> list[tuple[list[Pair], list[int], int]]:
    """Build reproducible demo-order variants for one synthetic prompt.

    A value of 1 means original order only. A value of 2 means original order
    plus one seeded shuffle, matching the paper-style "n = 2" idea.
    """
    if num_demo_permutations < 1:
        raise ValueError("--num-demo-permutations must be greater than 0")

    variants = [(demo_pairs, demo_indices, 0)]
    if len(demo_pairs) < 2:
        return variants

    for permutation_index in range(1, num_demo_permutations):
        order = list(range(len(demo_pairs)))
        variant_seed = seed + sum(ord(char) for char in task_id)
        variant_seed += heldout_index * 101 + permutation_index * 1009
        variant_seed += sum(ord(char) for char in transform_name)
        random.Random(variant_seed).shuffle(order)

        variants.append(
            (
                [demo_pairs[pos] for pos in order],
                [demo_indices[pos] for pos in order],
                permutation_index,
            )
        )
    return variants


def build_ttt_records_for_task(
    task_id: str,
    selected_pairs: list[Pair],
    selected_indices: list[int],
    enable_loo: bool = True,
    enable_train_transforms: bool = True,
    num_demo_permutations: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create JSONL-ready training records for one original ARC task.

    With k selected demos and identity-only transforms:
    - LOO enabled creates k records.
    - LOO disabled creates k direct I/O records.

    With transforms enabled, each record is repeated for every transform. With
    permutations enabled, each transformed row is repeated with shuffled demo
    order variants.
    """
    transform_names = DEFAULT_TRANSFORMS if enable_train_transforms else ("identity",)
    records: list[dict[str, Any]] = []

    # Each selected demo takes one turn being the held-out synthetic query.
    for heldout_pos, heldout_pair in enumerate(selected_pairs):
        if enable_loo:
            # Leave-one-out: one known demo pretends to be the query, and the
            # remaining demos become the in-context examples.
            demo_pairs = [
                pair for pos, pair in enumerate(selected_pairs) if pos != heldout_pos
            ]
            demo_indices = [
                idx for pos, idx in enumerate(selected_indices) if pos != heldout_pos
            ]
            mode = "loo_icl"
        else:
            # Direct I/O ablation: train on x -> y without task demonstrations.
            demo_pairs = []
            demo_indices = []
            mode = "direct_io"

        # Transform augmentation multiplies the tiny per-task dataset while
        # preserving the underlying ARC rule.
        for transform_name in transform_names:
            # A transform must apply to every input and output in the synthetic
            # task, otherwise the rule relationship changes.
            transformed_demos = transform_pairs(demo_pairs, transform_name)
            transformed_query = transform_pair(heldout_pair, transform_name)

            order_variants = demo_order_variants(
                transformed_demos,
                demo_indices,
                num_demo_permutations=num_demo_permutations if enable_loo else 1,
                seed=seed,
                task_id=task_id,
                heldout_index=selected_indices[heldout_pos],
                transform_name=transform_name,
            )
            for ordered_demos, ordered_indices, permutation_index in order_variants:
                prompt = build_icl_prompt(ordered_demos, transformed_query["input"])
                target = build_target(transformed_query["output"])

                # One record becomes one supervised training row for LoRA.
                records.append(
                    {
                        "task_id": task_id,
                        "mode": mode,
                        "transform": transform_name,
                        "inverse_transform": INVERSE_TRANSFORMS[transform_name],
                        "demo_permutation_index": permutation_index,
                        "heldout_demo_index": selected_indices[heldout_pos],
                        "context_demo_indices": ordered_indices,
                        "prompt": prompt,
                        "target": target,
                    }
                )

    return records


def cap_records(
    records: list[dict[str, Any]],
    max_records: int | None,
    seed: int,
    task_id: str,
) -> list[dict[str, Any]]:
    """Limit one task's TTT dataset size, matching the paper's 250-row cap."""
    if max_records is None:
        return records
    if max_records < 1:
        raise ValueError("--max-ttt-records must be greater than 0")
    if len(records) <= max_records:
        return records

    # Use a task-specific seed so multi-task runs are reproducible without
    # selecting the exact same row positions for every task.
    task_seed = seed + sum(ord(char) for char in task_id)
    selected = random.Random(task_seed).sample(records, max_records)
    return sorted(
        selected,
        key=lambda row: (
            row["heldout_demo_index"],
            row["transform"],
            row["mode"],
        ),
    )


def prepare_ttt_records(
    input_folder: str | Path,
    k_train_examples: int,
    skip_on_insufficient_demos: bool,
    enable_loo: bool,
    enable_train_transforms: bool,
    max_ttt_records: int | None = 250,
    num_demo_permutations: int = 1,
    task_id: str | None = None,
    num_tasks: int | None = None,
    seed: int = 42,
    shuffle_demos: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare all TTT records for one or more selected ARC tasks."""
    # The manifest is intentionally explicit so every adapter artifact can be
    # traced back to task IDs, selected demo indices, flags, and seed.
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    selected_tasks: list[dict[str, Any]] = []

    # This loop supports both one-task debugging and N-task batch experiments.
    for task_path in select_task_paths(input_folder, task_id, num_tasks, seed):
        current_task_id = task_id_from_path(task_path)
        task = load_task(task_path)

        # Select the demonstrations that will be used for both ICL prompts and
        # task-specific adapter training.
        selection = select_demo_pairs(
            task,
            k_train_examples=k_train_examples,
            seed=seed,
            shuffle_demos=shuffle_demos,
            skip_on_insufficient_demos=skip_on_insufficient_demos,
        )
        if selection is None:
            # Skipped tasks still appear in the manifest so the run summary
            # explains why no training rows were created for them.
            skipped.append(
                {
                    "task_id": current_task_id,
                    "reason": "insufficient_demos",
                    "available_demos": len(task["train"]),
                    "requested_demos": k_train_examples,
                }
            )
            continue

        selected_pairs, selected_indices = selection

        # Expand selected demos into LOO/direct-I/O rows, then optional TF rows.
        task_records = build_ttt_records_for_task(
            task_id=current_task_id,
            selected_pairs=selected_pairs,
            selected_indices=selected_indices,
            enable_loo=enable_loo,
            enable_train_transforms=enable_train_transforms,
            num_demo_permutations=num_demo_permutations,
            seed=seed,
        )
        capped_task_records = cap_records(
            task_records,
            max_records=max_ttt_records,
            seed=seed,
            task_id=current_task_id,
        )

        # Record task-level metadata before appending rows to the full run.
        selected_tasks.append(
            {
                "task_id": current_task_id,
                "available_demos": len(task["train"]),
                "selected_demo_indices": selected_indices,
                "test_count": len(task["test"]),
                "generated_record_count": len(task_records),
                "used_record_count": len(capped_task_records),
            }
        )
        records.extend(capped_task_records)

    # The manifest is written beside the JSONL and should travel with any
    # future SageMaker adapter artifact.
    manifest = {
        "input_folder": str(input_folder),
        "k_train_examples": k_train_examples,
        "skip_on_insufficient_demos": skip_on_insufficient_demos,
        "enable_loo": enable_loo,
        "enable_train_transforms": enable_train_transforms,
        "max_ttt_records": max_ttt_records,
        "num_demo_permutations": num_demo_permutations,
        "seed": seed,
        "shuffle_demos": shuffle_demos,
        "record_count": len(records),
        "selected_tasks": selected_tasks,
        "skipped_tasks": skipped,
    }
    return records, manifest


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write training rows in JSONL format, one supervised sample per line."""
    output_path = Path(output_path)

    # mkdir mirrors SageMaker/local behavior: callers can pass a fresh output
    # folder and the script creates it.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(value: dict[str, Any], output_path: str | Path) -> None:
    """Write a human-readable JSON manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
