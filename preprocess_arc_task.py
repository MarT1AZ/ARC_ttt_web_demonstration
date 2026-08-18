"""Preprocess one ARC task into SageMaker-style train/validation/test channels.

This script is about task preparation, not model training:
- train channel: LOO + transform + permutation rows from task["train"]
- validation channel: duplicate of train for simple trainer plumbing in v1
- test channel: the real task["test"][test_index] prompt for later inference
- S3 upload and Feature Group metadata are optional switches
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ttt_data import (
    DEFAULT_TRANSFORMS,
    INVERSE_TRANSFORMS,
    build_icl_prompt,
    build_ttt_records_for_task,
    cap_records,
    load_task,
    select_demo_pairs,
    task_id_from_path,
    transform_pair,
    transform_pairs,
    write_json,
    write_jsonl,
)


DEFAULT_BUCKET = "arc-ttt-artifact"
DEFAULT_PREFIX = "task_ttt_prep"
DEFAULT_FEATURE_GROUP = "arc-adapterops-task-prep"


def str_to_bool(value: str) -> bool:
    """Parse true/false CLI values without argparse's bool surprises."""
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def utc_now_id() -> str:
    """Use one timestamp for run ID and Feature Store event time."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def s3_uri(bucket: str, key: str) -> str:
    """Format an S3 URI from bucket and object key."""
    return f"s3://{bucket}/{key}"


def build_test_records(
    task_id: str,
    selected_pairs: list[dict[str, Any]],
    selected_indices: list[int],
    test_pairs: list[dict[str, Any]],
    enable_test_transforms: bool,
) -> list[dict[str, Any]]:
    """Create final inference/eval prompts from task["test"].

    v1 normally writes one identity row. If test transforms are enabled later,
    each row carries inverse_transform so evaluation can map predictions back.
    """
    transform_names = DEFAULT_TRANSFORMS if enable_test_transforms else ("identity",)
    records: list[dict[str, Any]] = []

    for test_index, test_pair in enumerate(test_pairs):
        for transform_name in transform_names:
            transformed_demos = transform_pairs(selected_pairs, transform_name)
            transformed_test = transform_pair(test_pair, transform_name)
            records.append(
                {
                    "task_id": task_id,
                    "channel": "test",
                    "test_index": test_index,
                    "transform": transform_name,
                    "inverse_transform": INVERSE_TRANSFORMS[transform_name],
                    "context_demo_indices": selected_indices,
                    "prompt": build_icl_prompt(transformed_demos, transformed_test["input"]),
                    "expected_output": transformed_test.get("output"),
                    "original_expected_output": deepcopy(test_pair.get("output")),
                }
            )

    return records


def write_task_split(
    output_folder: Path,
    task_id: str,
    task_path: Path,
    selected_pairs: list[dict[str, Any]],
    selected_indices: list[int],
    test_pairs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    """Write the source-of-truth split file with actual grids included."""
    split_path = output_folder / "task_split.json"
    write_json(
        {
            "task_id": task_id,
            "source_task_path": str(task_path),
            "k_train_examples": args.k_train_examples,
            "selected_demo_indices": selected_indices,
            "selected_demos": selected_pairs,
            "test_pairs": test_pairs,
            "seed": args.seed,
            "shuffle_demos": args.shuffle_demos,
        },
        split_path,
    )
    return split_path


def upload_folder_to_s3(local_folder: Path, bucket: str, prefix: str, region: str) -> dict[str, str]:
    """Upload every generated artifact and return local relative path -> S3 URI."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    uploaded: dict[str, str] = {}
    for path in local_folder.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(local_folder).as_posix()
        key = f"{prefix.rstrip('/')}/{relative}"
        s3.upload_file(str(path), bucket, key)
        uploaded[relative] = s3_uri(bucket, key)
    return uploaded


def feature_group_exists(sm_client, feature_group_name: str) -> bool:
    """Return true if the Feature Group already exists."""
    from botocore.exceptions import ClientError

    try:
        sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}:
            return False
        raise


def create_feature_group(sm_client, feature_group_name: str) -> None:
    """Create the online-only Feature Group for task prep metadata."""
    sm_client.create_feature_group(
        FeatureGroupName=feature_group_name,
        RecordIdentifierFeatureName="record_id",
        EventTimeFeatureName="event_time",
        FeatureDefinitions=[
            {"FeatureName": "record_id", "FeatureType": "String"},
            {"FeatureName": "event_time", "FeatureType": "String"},
            {"FeatureName": "task_id", "FeatureType": "String"},
            {"FeatureName": "run_id", "FeatureType": "String"},
            {"FeatureName": "s3_bucket", "FeatureType": "String"},
            {"FeatureName": "s3_prefix", "FeatureType": "String"},
            {"FeatureName": "task_split_s3_uri", "FeatureType": "String"},
            {"FeatureName": "manifest_s3_uri", "FeatureType": "String"},
            {"FeatureName": "train_s3_uri", "FeatureType": "String"},
            {"FeatureName": "validation_s3_uri", "FeatureType": "String"},
            {"FeatureName": "test_s3_uri", "FeatureType": "String"},
            {"FeatureName": "k_train_examples", "FeatureType": "Integral"},
            {"FeatureName": "max_ttt_records", "FeatureType": "Integral"},
            {"FeatureName": "num_demo_permutations", "FeatureType": "Integral"},
            {"FeatureName": "train_record_count", "FeatureType": "Integral"},
            {"FeatureName": "validation_record_count", "FeatureType": "Integral"},
            {"FeatureName": "test_record_count", "FeatureType": "Integral"},
            {"FeatureName": "test_index", "FeatureType": "Integral"},
            {"FeatureName": "enable_loo", "FeatureType": "String"},
            {"FeatureName": "enable_train_transforms", "FeatureType": "String"},
            {"FeatureName": "enable_test_transforms", "FeatureType": "String"},
            {"FeatureName": "selected_demo_indices", "FeatureType": "String"},
        ],
        OnlineStoreConfig={"EnableOnlineStore": True},
        Description="ARC AdapterOps task preprocessing metadata.",
        Tags=[{"Key": "project", "Value": "arc-adapterops"}],
    )


def wait_for_feature_group_created(
    sm_client,
    feature_group_name: str,
    timeout_seconds: int,
    poll_seconds: int = 10,
) -> None:
    """Wait until Feature Group is ready for put_record."""
    deadline = time.time() + timeout_seconds
    while True:
        desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = desc.get("FeatureGroupStatus")
        if status == "Created":
            return
        if status == "CreateFailed":
            reason = desc.get("FailureReason", "unknown")
            raise RuntimeError(f"Feature Group creation failed: {reason}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Feature Group: {feature_group_name}")
        print(f"Feature Group status: {status}; waiting {poll_seconds}s")
        time.sleep(poll_seconds)


def put_feature_record(
    featurestore_client,
    feature_group_name: str,
    record: dict[str, Any],
) -> None:
    """Write one metadata record to SageMaker Feature Store."""
    featurestore_client.put_record(
        FeatureGroupName=feature_group_name,
        Record=[
            {"FeatureName": name, "ValueAsString": str(value)}
            for name, value in record.items()
        ],
        TargetStores=["OnlineStore"],
    )


def build_parser() -> argparse.ArgumentParser:
    """Declare preprocessing args for local and SageMaker Studio runs."""
    parser = argparse.ArgumentParser(
        description="Preprocess one ARC task into train/validation/test channels."
    )
    parser.add_argument("--input-task-path", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--k-train-examples", type=int, default=4)
    parser.add_argument("--skip-on-insufficient-demos", type=str_to_bool, default=True)
    parser.add_argument("--shuffle-demos", type=str_to_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enable-loo", type=str_to_bool, default=True)
    parser.add_argument("--enable-train-transforms", type=str_to_bool, default=True)
    parser.add_argument("--enable-test-transforms", type=str_to_bool, default=False)
    parser.add_argument("--num-demo-permutations", type=int, default=2)
    parser.add_argument("--max-ttt-records", type=int, default=250)

    parser.add_argument("--upload-s3", type=str_to_bool, default=False)
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--region", default="ap-southeast-1")

    parser.add_argument("--write-feature-record", type=str_to_bool, default=False)
    parser.add_argument("--feature-group-name", default=DEFAULT_FEATURE_GROUP)
    parser.add_argument("--feature-group-timeout-seconds", type=int, default=900)
    return parser


def main() -> int:
    """Run task preprocessing and optional AWS handoff."""
    args = build_parser().parse_args()
    run_id = args.run_id or utc_now_id()

    input_task_path = Path(args.input_task_path)
    task_id = task_id_from_path(input_task_path)
    task = load_task(input_task_path)
    output_folder = Path(args.output_folder)

    selection = select_demo_pairs(
        task,
        k_train_examples=args.k_train_examples,
        seed=args.seed,
        shuffle_demos=args.shuffle_demos,
        skip_on_insufficient_demos=args.skip_on_insufficient_demos,
    )
    if selection is None:
        raise ValueError(
            f"Task {task_id} has {len(task['train'])} demos, "
            f"but k={args.k_train_examples}; skipped by policy."
        )
    selected_pairs, selected_indices = selection
    test_pairs = deepcopy(task["test"])

    train_records = build_ttt_records_for_task(
        task_id=task_id,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
        enable_loo=args.enable_loo,
        enable_train_transforms=args.enable_train_transforms,
        num_demo_permutations=args.num_demo_permutations,
        seed=args.seed,
    )
    train_records = cap_records(
        train_records,
        max_records=args.max_ttt_records,
        seed=args.seed,
        task_id=task_id,
    )

    # v1 duplicates validation for channel plumbing. This keeps the trainer
    # simple: both train and validation are prompt->target JSONL rows.
    validation_records = [dict(record, channel="validation") for record in train_records]
    train_records = [dict(record, channel="train") for record in train_records]
    test_records = build_test_records(
        task_id=task_id,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
        test_pairs=test_pairs,
        enable_test_transforms=args.enable_test_transforms,
    )

    split_path = write_task_split(
        output_folder=output_folder,
        task_id=task_id,
        task_path=input_task_path,
        selected_pairs=selected_pairs,
        selected_indices=selected_indices,
        test_pairs=test_pairs,
        args=args,
    )
    train_path = output_folder / "train" / "ttt_train.jsonl"
    validation_path = output_folder / "validation" / "ttt_validation.jsonl"
    test_path = output_folder / "test" / "test.jsonl"
    manifest_path = output_folder / "manifest.json"

    write_jsonl(train_records, train_path)
    write_jsonl(validation_records, validation_path)
    write_jsonl(test_records, test_path)

    manifest = {
        "task_id": task_id,
        "run_id": run_id,
        "input_task_path": str(input_task_path),
        "output_folder": str(output_folder),
        "k_train_examples": args.k_train_examples,
        "selected_demo_indices": selected_indices,
        "test_count": len(test_pairs),
        "enable_loo": args.enable_loo,
        "enable_train_transforms": args.enable_train_transforms,
        "enable_test_transforms": args.enable_test_transforms,
        "num_demo_permutations": args.num_demo_permutations,
        "max_ttt_records": args.max_ttt_records,
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "test_record_count": len(test_records),
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix,
    }
    write_json(manifest, manifest_path)

    s3_prefix = f"{args.s3_prefix.rstrip('/')}/{task_id}/{run_id}"
    uploaded = {}
    if args.upload_s3:
        uploaded = upload_folder_to_s3(
            local_folder=output_folder,
            bucket=args.s3_bucket,
            prefix=s3_prefix,
            region=args.region,
        )
        print(f"Uploaded artifacts to {s3_uri(args.s3_bucket, s3_prefix + '/')}")

    if args.write_feature_record:
        if not args.upload_s3:
            raise ValueError("--write-feature-record true requires --upload-s3 true")

        import boto3

        sm_client = boto3.client("sagemaker", region_name=args.region)
        if feature_group_exists(sm_client, args.feature_group_name):
            raise RuntimeError(
                f"Feature Group already exists: {args.feature_group_name}. "
                "Delete it manually first if you want to recreate it."
            )

        create_feature_group(sm_client, args.feature_group_name)
        wait_for_feature_group_created(
            sm_client,
            args.feature_group_name,
            timeout_seconds=args.feature_group_timeout_seconds,
        )

        event_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {
            "record_id": f"{task_id}-{run_id}",
            "event_time": event_time,
            "task_id": task_id,
            "run_id": run_id,
            "s3_bucket": args.s3_bucket,
            "s3_prefix": s3_prefix,
            "task_split_s3_uri": uploaded.get("task_split.json", ""),
            "manifest_s3_uri": uploaded.get("manifest.json", ""),
            "train_s3_uri": uploaded.get("train/ttt_train.jsonl", ""),
            "validation_s3_uri": uploaded.get("validation/ttt_validation.jsonl", ""),
            "test_s3_uri": uploaded.get("test/test.jsonl", ""),
            "k_train_examples": args.k_train_examples,
            "max_ttt_records": args.max_ttt_records,
            "num_demo_permutations": args.num_demo_permutations,
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "test_record_count": len(test_records),
            "enable_loo": str(args.enable_loo).lower(),
            "enable_train_transforms": str(args.enable_train_transforms).lower(),
            "enable_test_transforms": str(args.enable_test_transforms).lower(),
            "selected_demo_indices": json.dumps(selected_indices),
        }
        fs_client = boto3.client("sagemaker-featurestore-runtime", region_name=args.region)
        put_feature_record(fs_client, args.feature_group_name, record)
        print(f"Wrote Feature Group record: {record['record_id']}")

    print(f"Task split: {split_path}")
    print(f"Train channel: {train_path} ({len(train_records)} rows)")
    print(f"Validation channel: {validation_path} ({len(validation_records)} rows)")
    print(f"Test channel: {test_path} ({len(test_records)} rows)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
