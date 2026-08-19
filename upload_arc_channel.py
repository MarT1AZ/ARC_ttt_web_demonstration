"""Upload one ARC data channel to S3 and create a Feature Group.

This script intentionally does only the first storage/metadata step:

1. Abort if the Feature Group already exists.
2. Create the Feature Group.
3. Upload local ARC JSON files to S3.
4. Verify every uploaded S3 object exists.

It does not preprocess ARC tasks and it does not train a model.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def print_step(step: int, title: str) -> None:
    """Print visible step boundaries for notebook/terminal validation."""
    print("")
    print(f"STEP {step}: {title}")
    print("-" * (8 + len(title)))


def s3_uri(bucket: str, key: str) -> str:
    """Build an s3:// URI from a bucket and key."""
    return f"s3://{bucket}/{key}"


def build_s3_key(save_prefix: str, channel_name: str, relative_path: str) -> str:
    """Build a clean S3 key even when save_prefix is empty."""
    parts = [save_prefix.strip("/"), channel_name.strip("/"), relative_path.strip("/")]
    return "/".join(part for part in parts if part)


def list_local_files(data_path: Path) -> list[Path]:
    """Return local files to upload.

    If data_path is a folder, upload every file below it.
    If data_path is one file, upload just that file.
    """
    if data_path.is_file():
        return [data_path]
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    files = sorted(path for path in data_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files found under data path: {data_path}")
    return files


def feature_group_exists(sagemaker_client, feature_group_name: str) -> bool:
    """Check whether the requested Feature Group already exists."""
    from botocore.exceptions import ClientError

    try:
        sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}:
            return False
        raise


def create_arc_upload_feature_group(
    sagemaker_session,
    feature_group_name: str,
    role_arn: str,
) -> None:
    """Create an online Feature Group through the SageMaker SDK session."""
    from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
    from sagemaker.feature_store.feature_group import FeatureGroup

    # Keep this Feature Group tiny: it is only metadata for the uploaded ARC
    # channel, not the ARC examples themselves.
    feature_definitions = [
        FeatureDefinition(feature_name="record_id", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="channel_name", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="bucket", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="save_prefix", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="s3_channel_uri", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="file_count", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="total_bytes", feature_type=FeatureTypeEnum.INTEGRAL),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name,
        feature_definitions=feature_definitions,
        sagemaker_session=sagemaker_session,
    )
    feature_group.create(
        s3_uri=False,
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=role_arn,
        enable_online_store=True,
        description="Metadata for uploaded ARC data channels.",
        tags=[{"Key": "project", "Value": "arc-adapterops"}],
    )


def resolve_role_arn(sagemaker_module, sagemaker_session, role_arn: str | None) -> str:
    """Use CLI role when provided, otherwise use the current Studio role."""
    if role_arn:
        return role_arn

    try:
        return sagemaker_module.get_execution_role(sagemaker_session=sagemaker_session)
    except Exception as exc:
        raise RuntimeError(
            "Could not auto-detect the SageMaker execution role. "
            "Run again with --role-arn arn:aws:iam::<account-id>:role/<role-name>."
        ) from exc


def wait_for_feature_group_created(
    sagemaker_client,
    feature_group_name: str,
    timeout_seconds: int,
    poll_seconds: int = 10,
) -> None:
    """Wait until the Feature Group is ready."""
    deadline = time.time() + timeout_seconds
    while True:
        desc = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = desc.get("FeatureGroupStatus")
        if status == "Created":
            return
        if status == "CreateFailed":
            raise RuntimeError(desc.get("FailureReason", "Feature Group creation failed"))
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Feature Group: {feature_group_name}")
        print(f"Feature Group status: {status}; waiting {poll_seconds}s")
        time.sleep(poll_seconds)


def build_upload_plan(
    files: list[Path],
    data_path: Path,
    bucket: str,
    save_prefix: str,
    channel_name: str,
) -> list[dict[str, Any]]:
    """Map each local file to its destination S3 key."""
    plan = []
    for file_path in files:
        # Preserve folder shape when uploading a directory.
        relative = file_path.name if data_path.is_file() else file_path.relative_to(data_path).as_posix()
        key = build_s3_key(save_prefix, channel_name, relative)
        plan.append({"local_path": file_path, "bucket": bucket, "key": key})
    return plan


def abort_if_s3_objects_exist(s3_client, upload_plan: list[dict[str, Any]]) -> None:
    """Abort if any destination key already exists."""
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
        sample = "\n".join(existing[:10])
        raise RuntimeError(
            "Abort: S3 destination already contains file(s) with the same name:\n"
            f"{sample}"
        )


def upload_and_verify(s3_client, upload_plan: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Upload files and verify every destination object exists."""
    total_bytes = 0
    for item in upload_plan:
        local_path = item["local_path"]
        total_bytes += local_path.stat().st_size
        s3_client.upload_file(str(local_path), item["bucket"], item["key"])

        # Verification is merged with upload: immediately confirm the object is
        # visible at the expected key.
        s3_client.head_object(Bucket=item["bucket"], Key=item["key"])

    first = upload_plan[0]
    channel_prefix = "/".join(first["key"].split("/")[:-1])
    return s3_uri(first["bucket"], channel_prefix + "/"), len(upload_plan), total_bytes


def build_parser() -> argparse.ArgumentParser:
    """Define the minimal upload args requested for this step."""
    parser = argparse.ArgumentParser(description="Upload one ARC data channel to S3.")
    parser.add_argument("--training-channel", default="training")
    parser.add_argument("--path-to-data", required=True)
    parser.add_argument("--save-prefix", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--feature-name", "--feature-group-name", dest="feature_group_name", required=True)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--feature-group-timeout-seconds", type=int, default=900)
    return parser


def main() -> int:
    """Run the upload workflow as explicit validation steps."""
    args = build_parser().parse_args()
    data_path = Path(args.path_to_data)

    # Preflight local data before touching AWS, so a bad path does not create a
    # Feature Group that the user must delete manually.
    files = list_local_files(data_path)

    import boto3
    import sagemaker

    # Use one shared boto/SageMaker session so Studio profile, region, and role
    # resolution behave the same way as later SageMaker training jobs.
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sagemaker_client = boto_session.client("sagemaker")
    s3_client = boto_session.client("s3")
    role_arn = resolve_role_arn(sagemaker, sagemaker_session, args.role_arn)

    # STEP 1: Fail fast if the metadata registry name is already taken.
    print_step(1, "Verify Feature Group does not already exist")
    if feature_group_exists(sagemaker_client, args.feature_group_name):
        raise RuntimeError(
            f"Abort: Feature Group already exists: {args.feature_group_name}. "
            "Delete it manually before recreating."
        )
    print(f"Feature Group is available: {args.feature_group_name}")

    # STEP 2: Create the empty Feature Group that will represent this channel.
    print_step(2, "Create Feature Group")
    create_arc_upload_feature_group(
        sagemaker_session=sagemaker_session,
        feature_group_name=args.feature_group_name,
        role_arn=role_arn,
    )
    wait_for_feature_group_created(
        sagemaker_client,
        args.feature_group_name,
        timeout_seconds=args.feature_group_timeout_seconds,
    )
    print(f"Feature Group created: {args.feature_group_name}")

    # STEP 3: Upload the requested ARC channel, but abort before upload if any
    # destination object already exists.
    print_step(3, "Upload data to S3")
    upload_plan = build_upload_plan(
        files=files,
        data_path=data_path,
        bucket=args.bucket,
        save_prefix=args.save_prefix,
        channel_name=args.training_channel,
    )
    abort_if_s3_objects_exist(s3_client, upload_plan)
    s3_path, file_count, total_bytes = upload_and_verify(s3_client, upload_plan)
    print(f"S3 path: {s3_path}")
    print(f"Uploaded files: {file_count}")
    print(f"Uploaded bytes: {total_bytes}")

    # STEP 4: Upload verification is already done by head_object after each
    # file upload; this final line makes the successful S3 path easy to copy.
    print_step(4, "Verify upload complete")
    print(f"Verified all uploaded objects under: {s3_path}")
    print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
