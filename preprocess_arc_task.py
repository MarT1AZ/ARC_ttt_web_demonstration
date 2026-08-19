"""Preprocess one ARC task and register the prep run in Feature Store.

This script owns the task-prep step only:

STEP 1: Ensure the Feature Group is ready.
STEP 2: Create local ARC train/test prep files.
STEP 3: Upload those files to exact S3 object paths.
STEP 4: Write one Feature Store record with paths and preprocessing args.
STEP 5: Read the record back and verify the run is complete.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Grid = list[list[int]]
Pair = dict[str, Grid]
Transform = Callable[[Grid], Grid]

# These transforms create small, invertible ARC variants for TTT data.
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
    """Load one ARC task file and fail early if it is not ARC-shaped."""
    with Path(task_path).open("r", encoding="utf-8") as f:
        task = json.load(f)
    if "train" not in task or "test" not in task:
        raise ValueError(f"ARC task is missing train/test keys: {task_path}")
    return task


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
    """Select k task demonstrations and keep their original task indices."""
    if k_train_examples < 1:
        raise ValueError("--k-train-examples must be greater than 0")

    demos = task["train"]
    if len(demos) < k_train_examples and skip_on_insufficient_demos:
        return None

    indices = list(range(len(demos)))
    if shuffle_demos:
        # Seeded shuffle keeps runs reproducible across SageMaker reruns.
        random.Random(seed).shuffle(indices)

    selected_indices = indices[: min(k_train_examples, len(indices))]
    selected = [deepcopy(demos[i]) for i in selected_indices]
    return selected, selected_indices


def grid_to_text(grid: Grid) -> str:
    """Serialize an ARC grid compactly for prompt text."""
    return json.dumps(grid, separators=(",", ":"))


def build_icl_prompt(demo_pairs: list[Pair], query_grid: Grid) -> str:
    """Build the prompt shared by ICL test rows and LOO training rows."""
    lines = [
        "You are solving an ARC grid transformation task.",
        "Infer the rule from the examples, then answer the query.",
        "Return only JSON in this exact format: {\"output\": [[...]]}",
        "",
    ]

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

    lines.extend(["Query input:", grid_to_text(query_grid), "Answer:"])
    return "\n".join(lines)


def build_target(output_grid: Grid) -> str:
    """Build the supervised target text for one JSONL training row."""
    return json.dumps({"output": output_grid}, separators=(",", ":"))


def identity(grid: Grid) -> Grid:
    """No-op transform."""
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
    """Flip grid horizontally."""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Flip grid vertically."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Reflect grid across the main diagonal."""
    return [list(row) for row in zip(*grid)]


TRANSFORMS: dict[str, Transform] = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
}

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
    """Apply one transform to both input and output grids."""
    fn = TRANSFORMS[transform_name]
    return {"input": fn(pair["input"]), "output": fn(pair["output"])}


def transform_pairs(pairs: list[Pair], transform_name: str) -> list[Pair]:
    """Apply one transform to a list of ARC pairs."""
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
    """Create reproducible prompt-order variants for one synthetic row."""
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
    """Create prompt/target TTT records for one ARC task."""
    transform_names = DEFAULT_TRANSFORMS if enable_train_transforms else ("identity",)
    records: list[dict[str, Any]] = []

    for heldout_pos, heldout_pair in enumerate(selected_pairs):
        if enable_loo:
            # One demo becomes the supervised query; the rest become context.
            demo_pairs = [
                pair for pos, pair in enumerate(selected_pairs) if pos != heldout_pos
            ]
            demo_indices = [
                idx for pos, idx in enumerate(selected_indices) if pos != heldout_pos
            ]
            mode = "loo_icl"
        else:
            # Direct I/O ablation has no context examples.
            demo_pairs = []
            demo_indices = []
            mode = "direct_io"

        for transform_name in transform_names:
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
                # The row keeps prompt/target for training and indices for audit.
                records.append(
                    {
                        "task_id": task_id,
                        "mode": mode,
                        "transform": transform_name,
                        "inverse_transform": INVERSE_TRANSFORMS[transform_name],
                        "demo_permutation_index": permutation_index,
                        "heldout_demo_index": selected_indices[heldout_pos],
                        "context_demo_indices": ordered_indices,
                        "prompt": build_icl_prompt(
                            ordered_demos,
                            transformed_query["input"],
                        ),
                        "target": build_target(transformed_query["output"]),
                    }
                )

    return records


def cap_records(
    records: list[dict[str, Any]],
    max_records: int | None,
    seed: int,
    task_id: str,
) -> list[dict[str, Any]]:
    """Limit one task's generated rows with a reproducible sample."""
    if max_records is None:
        return records
    if max_records < 1:
        raise ValueError("--max-ttt-records must be greater than 0")
    if len(records) <= max_records:
        return records

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


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write JSONL records and create the output folder if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(value: dict[str, Any], output_path: str | Path) -> None:
    """Write a readable JSON file and create the output folder if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


DEFAULT_BUCKET = "arc-ttt-artifact"
DEFAULT_PREFIX = "ttt_data"
DEFAULT_FEATURE_GROUP = "arc_adapterops_task_prep"

# Feature Group schema for processed ARC prep runs. One record = one run.
REQUIRED_FEATURES = {
    "record_id",
    "event_time",
    "task_id",
    "run_id",
    "dataset_split",
    "training_s3_uri",
    "test_s3_uri",
    "manifest_s3_uri",
    "task_split_s3_uri",
    "generated_ttt_rows",
    "train_rows",
    "was_capped",
    "test_rows",
    "k_train_examples",
    "skip_on_insufficient_demos",
    "shuffle_demos",
    "seed",
    "enable_loo",
    "enable_train_transforms",
    "enable_test_transforms",
    "num_demo_permutations",
    "max_ttt_records",
    "selected_demo_indices",
}


def str_to_bool(value: str) -> bool:
    """Parse true/false CLI values without argparse's bool surprises."""
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def utc_now_id() -> str:
    """Build a stable run ID for this preprocessing run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def event_time_now() -> str:
    """Build the Feature Store event_time value."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bool_text(value: bool) -> str:
    """Store booleans as lowercase strings in Feature Store."""
    return str(value).lower()


def s3_uri(bucket: str, key: str) -> str:
    """Format an S3 URI from bucket and object key."""
    return f"s3://{bucket}/{key}"


def build_s3_key(*parts: str) -> str:
    """Join S3 key parts while allowing empty prefixes."""
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def print_step(step: int, title: str) -> None:
    """Print strict step sections for notebook/terminal debugging."""
    print("")
    print(f"STEP {step}: {title}")
    print("-" * (8 + len(title)))


def infer_dataset_split(task_path: Path, explicit_split: str | None) -> str:
    """Use the CLI split if provided, otherwise infer from parent folder."""
    if explicit_split:
        return explicit_split
    return task_path.parent.name or "unknown"


def resolve_role_arn(args: argparse.Namespace) -> str:
    """Use CLI role when provided, otherwise try the current Studio role."""
    if args.role_arn:
        return args.role_arn

    try:
        from sagemaker.core.helper.session_helper import get_execution_role

        return get_execution_role()
    except Exception:
        pass

    try:
        import sagemaker

        return sagemaker.get_execution_role()
    except Exception as exc:
        raise RuntimeError(
            "Abort: could not auto-detect role ARN for Feature Store offline storage. "
            "Pass --role-arn arn:aws:iam::<account-id>:role/<role-name>."
        ) from exc


def feature_definitions() -> list[dict[str, str]]:
    """Return the prep Feature Group schema."""
    return [
        {"FeatureName": "record_id", "FeatureType": "String"},
        {"FeatureName": "event_time", "FeatureType": "String"},
        {"FeatureName": "task_id", "FeatureType": "String"},
        {"FeatureName": "run_id", "FeatureType": "String"},
        {"FeatureName": "dataset_split", "FeatureType": "String"},
        {"FeatureName": "training_s3_uri", "FeatureType": "String"},
        {"FeatureName": "test_s3_uri", "FeatureType": "String"},
        {"FeatureName": "manifest_s3_uri", "FeatureType": "String"},
        {"FeatureName": "task_split_s3_uri", "FeatureType": "String"},
        {"FeatureName": "generated_ttt_rows", "FeatureType": "Integral"},
        {"FeatureName": "train_rows", "FeatureType": "Integral"},
        {"FeatureName": "was_capped", "FeatureType": "String"},
        {"FeatureName": "test_rows", "FeatureType": "Integral"},
        {"FeatureName": "k_train_examples", "FeatureType": "Integral"},
        {"FeatureName": "skip_on_insufficient_demos", "FeatureType": "String"},
        {"FeatureName": "shuffle_demos", "FeatureType": "String"},
        {"FeatureName": "seed", "FeatureType": "Integral"},
        {"FeatureName": "enable_loo", "FeatureType": "String"},
        {"FeatureName": "enable_train_transforms", "FeatureType": "String"},
        {"FeatureName": "enable_test_transforms", "FeatureType": "String"},
        {"FeatureName": "num_demo_permutations", "FeatureType": "Integral"},
        {"FeatureName": "max_ttt_records", "FeatureType": "Integral"},
        {"FeatureName": "selected_demo_indices", "FeatureType": "String"},
    ]


def create_feature_group(sm_client, args: argparse.Namespace) -> None:
    """Create the prep Feature Group when first-run setup is allowed."""
    offline_store_uri = s3_uri(args.s3_bucket, build_s3_key(args.s3_prefix, "feature-store"))

    sm_client.create_feature_group(
        FeatureGroupName=args.feature_group_name,
        RecordIdentifierFeatureName="record_id",
        EventTimeFeatureName="event_time",
        FeatureDefinitions=feature_definitions(),
        OnlineStoreConfig={"EnableOnlineStore": True},
        OfflineStoreConfig={
            "S3StorageConfig": {"S3Uri": offline_store_uri},
            "DisableGlueTableCreation": False,
        },
        RoleArn=resolve_role_arn(args),
        Description="ARC AdapterOps task preprocessing metadata.",
        Tags=[{"Key": "project", "Value": "arc-adapterops"}],
    )


def wait_for_feature_group_created(
    sm_client,
    feature_group_name: str,
    timeout_seconds: int,
    poll_seconds: int = 10,
) -> None:
    """Wait until a newly created Feature Group is ready."""
    deadline = time.time() + timeout_seconds
    while True:
        desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = desc.get("FeatureGroupStatus")
        if status == "Created":
            return
        if status == "CreateFailed":
            raise RuntimeError(desc.get("FailureReason", "Feature Group creation failed"))
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Feature Group: {feature_group_name}")
        print(f"Feature Group status: {status}; waiting {poll_seconds}s")
        time.sleep(poll_seconds)


def build_test_records(
    task_id: str,
    selected_pairs: list[dict[str, Any]],
    selected_indices: list[int],
    test_pairs: list[dict[str, Any]],
    enable_test_transforms: bool,
) -> list[dict[str, Any]]:
    """Create generic task['test'] rows for downstream evaluation."""
    transform_names = DEFAULT_TRANSFORMS if enable_test_transforms else ("identity",)
    records: list[dict[str, Any]] = []

    for test_index, test_pair in enumerate(test_pairs):
        for transform_name in transform_names:
            # Transformed test rows keep inverse_transform so downstream code can
            # map predictions back before scoring.
            transformed_test = transform_pair(test_pair, transform_name)
            transformed_demos = transform_pairs(selected_pairs, transform_name)

            records.append(
                {
                    "task_id": task_id,
                    "test_index": test_index,
                    "transform": transform_name,
                    "inverse_transform": INVERSE_TRANSFORMS[transform_name],
                    "context_demo_indices": selected_indices,
                    # Keep structured grids beside the prompt so the viewer and
                    # future trainers do not need to reverse-parse prompt text.
                    "context_pairs": transformed_demos,
                    "query_input": transformed_test["input"],
                    "target_output": transformed_test.get("output"),
                    "prompt": build_icl_prompt(transformed_demos, transformed_test["input"]),
                    "expected_output": transformed_test.get("output"),
                    "original_expected_output": deepcopy(test_pair.get("output")),
                }
            )

    return records


def add_training_grid_fields(
    train_records: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    """Attach structured grids to TTT rows inside the prep step."""
    pair_by_index = {
        selected_index: pair
        for selected_index, pair in zip(selected_indices, selected_pairs)
    }
    enriched_records: list[dict[str, Any]] = []

    for record in train_records:
        transform_name = record["transform"]
        context_pairs = [
            pair_by_index[index]
            for index in record.get("context_demo_indices", [])
        ]
        heldout_pair = pair_by_index[record["heldout_demo_index"]]

        # The prompt is still the trainer-ready text, but these grid fields make
        # the JSONL row inspectable and let future trainers build custom prompts.
        transformed_context = transform_pairs(context_pairs, transform_name)
        transformed_query = transform_pair(heldout_pair, transform_name)
        enriched_records.append(
            {
                **record,
                "context_pairs": transformed_context,
                "query_input": transformed_query["input"],
                "target_output": transformed_query["output"],
            }
        )

    return enriched_records


def step_1_ensure_feature_group_ready(args: argparse.Namespace) -> dict[str, Any]:
    """STEP 1: Create if allowed, then validate the prep Feature Group."""
    import boto3
    from botocore.exceptions import ClientError

    print_step(1, "Ensure Feature Group ready")
    sm_client = boto3.client("sagemaker", region_name=args.region)

    try:
        desc = sm_client.describe_feature_group(FeatureGroupName=args.feature_group_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}:
            if args.create_feature_group_if_missing:
                print(f"Feature Group missing; creating: {args.feature_group_name}")
                create_feature_group(sm_client, args)
                wait_for_feature_group_created(
                    sm_client,
                    args.feature_group_name,
                    timeout_seconds=args.feature_group_timeout_seconds,
                )
                desc = sm_client.describe_feature_group(FeatureGroupName=args.feature_group_name)
            else:
                raise RuntimeError(
                    f"Abort: Feature Group is missing: {args.feature_group_name}. "
                    "Pass --create-feature-group-if-missing true for first-time setup."
                ) from exc
        else:
            raise RuntimeError(
                f"Abort: could not describe Feature Group {args.feature_group_name}."
            ) from exc

    status = desc.get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Abort: Feature Group status is {status}, not Created.")

    existing_features = {item["FeatureName"] for item in desc.get("FeatureDefinitions", [])}
    missing_features = sorted(REQUIRED_FEATURES - existing_features)
    if missing_features:
        raise RuntimeError(
            "Abort: Feature Group schema is missing required field(s): "
            + ", ".join(missing_features)
        )

    if not desc.get("OnlineStoreConfig", {}).get("EnableOnlineStore"):
        raise RuntimeError("Abort: Feature Group must have Online Store enabled for get_record.")

    if "OfflineStoreConfig" not in desc:
        raise RuntimeError("Abort: Feature Group must have Offline Store enabled for history queries.")

    offline_uri = desc["OfflineStoreConfig"]["S3StorageConfig"]["S3Uri"]
    print(f"Feature Group: {args.feature_group_name}")
    print(f"Offline store: {offline_uri}")
    return {"sagemaker_client": sm_client, "feature_group_description": desc}


def step_2_create_arc_prep(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """STEP 2: Build local training and test files for one ARC task."""
    print_step(2, "Create ARC prep")

    input_task_path = Path(args.input_task_path)
    task_id = task_id_from_path(input_task_path)
    task = load_task(input_task_path)
    dataset_split = infer_dataset_split(input_task_path, args.dataset_split)

    selection = select_demo_pairs(
        task,
        k_train_examples=args.k_train_examples,
        seed=args.seed,
        shuffle_demos=args.shuffle_demos,
        skip_on_insufficient_demos=args.skip_on_insufficient_demos,
    )
    if selection is None:
        raise RuntimeError(
            f"Abort: task {task_id} has {len(task['train'])} demos, "
            f"but k={args.k_train_examples}."
        )

    selected_pairs, selected_indices = selection
    test_pairs = deepcopy(task["test"])

    # Training records are the TTT synthetic rows: LOO, optional transforms,
    # optional prompt-order permutations, then an optional row cap.
    train_records = build_ttt_records_for_task(
        task_id=task_id,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
        enable_loo=args.enable_loo,
        enable_train_transforms=args.enable_train_transforms,
        num_demo_permutations=args.num_demo_permutations,
        seed=args.seed,
    )
    generated_ttt_rows = len(train_records)
    train_records = cap_records(
        train_records,
        max_records=args.max_ttt_records,
        seed=args.seed,
        task_id=task_id,
    )
    training_records = add_training_grid_fields(
        train_records,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
    )

    test_records = build_test_records(
        task_id=task_id,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
        test_pairs=test_pairs,
        enable_test_transforms=args.enable_test_transforms,
    )

    output_folder = Path(args.output_folder)
    training_path = output_folder / "ttt_train.jsonl"
    test_path = output_folder / "test.jsonl"
    manifest_path = output_folder / "manifest.json"
    task_split_path = output_folder / "task_split.json"

    # The task split is the source-of-truth for which demos were selected.
    write_json(
        {
            "task_id": task_id,
            "run_id": run_id,
            "dataset_split": dataset_split,
            "source_task_path": str(input_task_path),
            "selected_demo_indices": selected_indices,
            "selected_demos": selected_pairs,
            "test_pairs": test_pairs,
        },
        task_split_path,
    )
    write_jsonl(training_records, training_path)
    write_jsonl(test_records, test_path)

    manifest = {
        "task_id": task_id,
        "run_id": run_id,
        "dataset_split": dataset_split,
        "input_task_path": str(input_task_path),
        "output_folder": str(output_folder),
        "training_path": str(training_path),
        "test_path": str(test_path),
        "task_split_path": str(task_split_path),
        "selected_demo_indices": selected_indices,
        "generated_ttt_rows": generated_ttt_rows,
        "train_rows": len(training_records),
        "was_capped": args.max_ttt_records is not None and generated_ttt_rows > len(training_records),
        "test_rows": len(test_records),
        "k_train_examples": args.k_train_examples,
        "skip_on_insufficient_demos": args.skip_on_insufficient_demos,
        "shuffle_demos": args.shuffle_demos,
        "seed": args.seed,
        "enable_loo": args.enable_loo,
        "enable_train_transforms": args.enable_train_transforms,
        "enable_test_transforms": args.enable_test_transforms,
        "num_demo_permutations": args.num_demo_permutations,
        "max_ttt_records": args.max_ttt_records,
    }
    write_json(manifest, manifest_path)

    print(f"Task ID: {task_id}")
    print(f"Generated TTT rows: {generated_ttt_rows}")
    print(f"Training rows: {len(training_records)}")
    print(f"Was capped: {manifest['was_capped']}")
    print(f"Test rows: {len(test_records)}")

    return {
        "task_id": task_id,
        "run_id": run_id,
        "dataset_split": dataset_split,
        "output_folder": output_folder,
        "training_path": training_path,
        "test_path": test_path,
        "manifest_path": manifest_path,
        "task_split_path": task_split_path,
        "manifest": manifest,
    }


def build_upload_plan(args: argparse.Namespace, prep: dict[str, Any]) -> list[dict[str, Any]]:
    """Map local prep files to exact S3 object keys in the run folder."""
    base_prefix = build_s3_key(args.s3_prefix, prep["task_id"], prep["run_id"])
    return [
        {
            "name": "training",
            "local_path": prep["training_path"],
            "bucket": args.s3_bucket,
            "key": build_s3_key(base_prefix, prep["training_path"].name),
        },
        {
            "name": "test",
            "local_path": prep["test_path"],
            "bucket": args.s3_bucket,
            "key": build_s3_key(base_prefix, prep["test_path"].name),
        },
        {
            "name": "manifest",
            "local_path": prep["manifest_path"],
            "bucket": args.s3_bucket,
            "key": build_s3_key(base_prefix, "manifest.json"),
        },
        {
            "name": "task_split",
            "local_path": prep["task_split_path"],
            "bucket": args.s3_bucket,
            "key": build_s3_key(base_prefix, "task_split.json"),
        },
    ]


def abort_if_s3_targets_exist(s3_client, upload_plan: list[dict[str, Any]]) -> None:
    """Abort before upload if any destination object already exists."""
    from botocore.exceptions import ClientError

    existing = []
    for item in upload_plan:
        try:
            s3_client.head_object(Bucket=item["bucket"], Key=item["key"])
            existing.append(s3_uri(item["bucket"], item["key"]))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                continue
            raise

    if existing:
        raise RuntimeError("Abort: S3 target already exists:\n" + "\n".join(existing[:20]))


def step_3_upload(args: argparse.Namespace, prep: dict[str, Any]) -> dict[str, str]:
    """STEP 3: Upload prep artifacts and verify each S3 object exists."""
    import boto3

    print_step(3, "Upload prep files to S3")
    s3_client = boto3.client("s3", region_name=args.region)
    upload_plan = build_upload_plan(args, prep)

    # This prevents accidental overwrite/reuse of a run_id.
    abort_if_s3_targets_exist(s3_client, upload_plan)

    uploaded_files: dict[str, str] = {}
    for item in upload_plan:
        s3_client.upload_file(str(item["local_path"]), item["bucket"], item["key"])
        s3_client.head_object(Bucket=item["bucket"], Key=item["key"])
        uploaded_files[item["name"]] = s3_uri(item["bucket"], item["key"])

    uploaded = {
        "training_s3_uri": uploaded_files["training"],
        "test_s3_uri": uploaded_files["test"],
        "manifest_s3_uri": uploaded_files["manifest"],
        "task_split_s3_uri": uploaded_files["task_split"],
    }

    print(f"Training file: {uploaded['training_s3_uri']}")
    print(f"Test file: {uploaded['test_s3_uri']}")
    print("Upload verified")
    return uploaded


def build_feature_record(
    args: argparse.Namespace,
    prep: dict[str, Any],
    uploaded: dict[str, str],
) -> dict[str, Any]:
    """Build one Feature Store record for this preprocessing run."""
    manifest = prep["manifest"]
    return {
        "record_id": f"{prep['task_id']}:{prep['run_id']}",
        "event_time": event_time_now(),
        "task_id": prep["task_id"],
        "run_id": prep["run_id"],
        "dataset_split": prep["dataset_split"],
        "training_s3_uri": uploaded["training_s3_uri"],
        "test_s3_uri": uploaded["test_s3_uri"],
        "manifest_s3_uri": uploaded["manifest_s3_uri"],
        "task_split_s3_uri": uploaded["task_split_s3_uri"],
        "generated_ttt_rows": manifest["generated_ttt_rows"],
        "train_rows": manifest["train_rows"],
        "was_capped": bool_text(manifest["was_capped"]),
        "test_rows": manifest["test_rows"],
        "k_train_examples": args.k_train_examples,
        "skip_on_insufficient_demos": bool_text(args.skip_on_insufficient_demos),
        "shuffle_demos": bool_text(args.shuffle_demos),
        "seed": args.seed,
        "enable_loo": bool_text(args.enable_loo),
        "enable_train_transforms": bool_text(args.enable_train_transforms),
        "enable_test_transforms": bool_text(args.enable_test_transforms),
        "num_demo_permutations": args.num_demo_permutations,
        "max_ttt_records": args.max_ttt_records,
        "selected_demo_indices": json.dumps(manifest["selected_demo_indices"]),
    }


def step_4_put_feature_record(
    args: argparse.Namespace,
    prep: dict[str, Any],
    uploaded: dict[str, str],
) -> dict[str, Any]:
    """STEP 4: Store S3 paths and preprocessing args in Feature Store."""
    import boto3

    print_step(4, "Put Feature Store record")
    record = build_feature_record(args, prep, uploaded)
    fs_client = boto3.client("sagemaker-featurestore-runtime", region_name=args.region)

    # Do not pass TargetStores. AWS writes to every store configured on the
    # Feature Group, which gives us Online lookup plus Offline history.
    fs_client.put_record(
        FeatureGroupName=args.feature_group_name,
        Record=[
            {"FeatureName": name, "ValueAsString": str(value)}
            for name, value in record.items()
        ],
    )

    print(f"Record ID: {record['record_id']}")
    return record


def step_5_verify_feature_record(args: argparse.Namespace, record: dict[str, Any]) -> None:
    """STEP 5: Read Feature Store online record and verify key paths."""
    import boto3
    from botocore.exceptions import ClientError

    print_step(5, "Verify Feature Store record")
    fs_client = boto3.client("sagemaker-featurestore-runtime", region_name=args.region)

    # Online Store readback can lag briefly after put_record, so retry before
    # declaring the prep run failed.
    response = None
    for attempt in range(1, 7):
        try:
            response = fs_client.get_record(
                FeatureGroupName=args.feature_group_name,
                RecordIdentifierValueAsString=record["record_id"],
            )
            if response.get("Record"):
                break
        except ClientError:
            if attempt == 6:
                raise
        time.sleep(5)

    if response is None or not response.get("Record"):
        raise RuntimeError("Abort: Feature Store readback returned no record.")

    readback = {
        item["FeatureName"]: item.get("ValueAsString", "")
        for item in response.get("Record", [])
    }

    for key in (
        "record_id",
        "training_s3_uri",
        "test_s3_uri",
        "manifest_s3_uri",
    ):
        if readback.get(key) != str(record[key]):
            raise RuntimeError(f"Abort: Feature Store readback mismatch for {key}.")

    print(f"Verified: {record['record_id']}")
    print("COMPLETE")


def build_parser() -> argparse.ArgumentParser:
    """Declare the ARC prep arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess one ARC task and register exact S3 object paths."
    )
    parser.add_argument("--input-task-path", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-split", default=None)

    parser.add_argument("--k-train-examples", type=int, default=4)
    parser.add_argument("--skip-on-insufficient-demos", type=str_to_bool, default=True)
    parser.add_argument("--shuffle-demos", type=str_to_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enable-loo", type=str_to_bool, default=True)
    parser.add_argument("--enable-train-transforms", type=str_to_bool, default=True)
    parser.add_argument("--enable-test-transforms", type=str_to_bool, default=False)
    parser.add_argument("--num-demo-permutations", type=int, default=2)
    parser.add_argument("--max-ttt-records", type=int, default=250)

    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--feature-group-name", default=DEFAULT_FEATURE_GROUP)
    parser.add_argument("--create-feature-group-if-missing", type=str_to_bool, default=False)
    parser.add_argument("--feature-group-timeout-seconds", type=int, default=900)
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--region", default="ap-southeast-1")
    return parser


def main() -> int:
    """Run the strict five-step preprocessing workflow."""
    args = build_parser().parse_args()
    run_id = args.run_id or utc_now_id()

    step_1_ensure_feature_group_ready(args)
    prep = step_2_create_arc_prep(args, run_id)
    uploaded = step_3_upload(args, prep)
    record = step_4_put_feature_record(args, prep, uploaded)
    step_5_verify_feature_record(args, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
