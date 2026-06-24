"""
AWS SageMaker Training Job Launcher.

Addresses [Vikram · Netflix]: "Your training timeout is 90 minutes. For a truly
massive dataset, you'd eventually outgrow GitHub Hosted Runners and need a
self-hosted GPU runner or an AWS SageMaker training job trigger."

Strategy
--------
This module:
  1. Measures the local feature store size.
  2. If it exceeds SAGEMAKER_THRESHOLD_GB, submits a SageMaker training job
     and waits for it to complete (or polls until done).
  3. If it's under threshold, signals the caller to train locally on the runner.

The SageMaker job:
  - Uses ml.p3.2xlarge (V100 GPU) Spot instances for cost efficiency (~70% saving).
  - Syncs feature data from S3, runs the same `python main.py --mode train`,
    then uploads the checkpoint back to S3.
  - Respects OIDC-sourced credentials (no static keys).

Exit codes:
  0 → SageMaker job completed successfully
  1 → SageMaker job failed
  2 → Dataset below threshold; caller should train locally
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


# ── Thresholds ────────────────────────────────────────────────────────────────

SAGEMAKER_THRESHOLD_GB = float(os.getenv("SAGEMAKER_THRESHOLD_GB", "5.0"))
DEFAULT_INSTANCE_TYPE  = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.p3.2xlarge")
DEFAULT_SPOT           = os.getenv("SAGEMAKER_USE_SPOT", "true").lower() == "true"
MAX_WAIT_HOURS         = int(os.getenv("SAGEMAKER_MAX_WAIT_HOURS", "6"))


# ── Dataset size helper ───────────────────────────────────────────────────────

def _estimate_zarr_inmemory_gb(zarr_path: Path) -> float:
    """Estimate the uncompressed in-memory tensor size of a single Zarr store in GiB.

    Reads the `.zarray` metadata files inside the Zarr directory to compute
    shape × dtype_itemsize for each array, which represents the true memory
    footprint when the data is loaded as numpy/torch tensors.

    This avoids the vulnerability where compressed or sparse Zarr chunks occupy
    very little disk space but expand to many GiB in RAM, causing OOM crashes
    when the launcher incorrectly decides to train locally.
    """
    import json as _json

    total_bytes = 0
    for zarray_file in zarr_path.rglob(".zarray"):
        try:
            with open(zarray_file) as f:
                meta = _json.load(f)
            shape = meta.get("shape", [])
            dtype_str = meta.get("dtype", "<f4")
            # numpy dtype string → itemsize (bytes per element)
            import numpy as _np
            itemsize = _np.dtype(dtype_str).itemsize
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            total_bytes += num_elements * itemsize
        except Exception:
            continue  # skip non-parseable metadata files

    return total_bytes / (1024 ** 3)


def _dataset_size_gb(features_dir: Path) -> float:
    """Return the estimated in-memory tensor size of all Zarr stores in GiB.

    Prefers the Zarr metadata-based estimate (uncompressed shape × dtype) over
    raw disk size, because compressed/sparse Zarr directories drastically
    under-report their true memory footprint on disk.

    Falls back to disk size only when no Zarr metadata is found (e.g. non-Zarr
    files in the directory).
    """
    zarr_stores = list(features_dir.glob("*.zarr"))
    if zarr_stores:
        inmemory_total = sum(_estimate_zarr_inmemory_gb(zs) for zs in zarr_stores)
        if inmemory_total > 0:
            disk_total = sum(
                f.stat().st_size for f in features_dir.rglob("*") if f.is_file()
            ) / (1024 ** 3)
            logger.info(
                f"Zarr in-memory estimate: {inmemory_total:.2f} GiB  "
                f"(disk size: {disk_total:.2f} GiB, "
                f"compression ratio: {disk_total / inmemory_total:.2f}x)"
            )
            return inmemory_total

    # Fallback: raw disk size for non-Zarr or metadata-less directories
    total = sum(
        f.stat().st_size
        for f in features_dir.rglob("*")
        if f.is_file()
    )
    return total / (1024 ** 3)


def _s3_dataset_size_gb(s3_bucket: str, s3_features_prefix: str) -> float:
    """Calculate the total size of processed Zarr files stored in S3 in GiB."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            "boto3 is required to check S3 dataset size. Install with: pip install boto3"
        )
    s3 = boto3.client("s3")
    total_bytes = 0
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_features_prefix):
        for obj in page.get('Contents', []):
            total_bytes += obj['Size']
    return total_bytes / (1024 ** 3)



# ── SageMaker launcher ────────────────────────────────────────────────────────

def _package_and_upload_code(s3_bucket: str, key_prefix: str) -> str:
    """Tar the local directory (main.py, src/, requirements.txt, configs/) and upload to S3.
    Returns the S3 URI of the uploaded source tarball.
    """
    import tarfile
    import tempfile
    import boto3
    
    s3 = boto3.client("s3")
    
    # Create a temporary file
    fd, temp_tar_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    
    try:
        logger.info(f"Packaging source code into tarball: {temp_tar_path}")
        with tarfile.open(temp_tar_path, "w:gz") as tar:
            # Add main.py
            if os.path.exists("main.py"):
                tar.add("main.py", arcname="main.py")
            # Add src directory
            if os.path.exists("src"):
                tar.add("src", arcname="src")
            # Add requirements.txt
            if os.path.exists("requirements.txt"):
                tar.add("requirements.txt", arcname="requirements.txt")
            # Add configs directory
            if os.path.exists("configs"):
                tar.add("configs", arcname="configs")
                
        s3_key = f"{key_prefix}/sourcedir.tar.gz"
        logger.info(f"Uploading packaged code to s3://{s3_bucket}/{s3_key}")
        s3.upload_file(temp_tar_path, s3_bucket, s3_key)
        return f"s3://{s3_bucket}/{s3_key}"
    finally:
        if os.path.exists(temp_tar_path):
            os.remove(temp_tar_path)


def launch_sagemaker_training(
    s3_bucket: str,
    s3_features_prefix: str,
    s3_output_prefix: str,
    role_arn: str,
    image_uri: Optional[str] = None,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    use_spot: bool = DEFAULT_SPOT,
    max_wait_hours: int = MAX_WAIT_HOURS,
    job_name_prefix: str = "crop-yield-train",
    no_wait: bool = False,
) -> Dict:
    """Submit a SageMaker training job and block until it finishes.

    Args:
        s3_bucket:          Bucket holding feature data and where checkpoints land.
        s3_features_prefix: S3 key prefix of the preprocessed feature Zarr stores.
        s3_output_prefix:   S3 key prefix where SageMaker writes the model output.
        role_arn:           IAM Role ARN for the SageMaker job (OIDC-assumed role).
        image_uri:          ECR image URI. Defaults to the official AWS PyTorch DLC.
        instance_type:      SageMaker instance type.
        use_spot:           Whether to use Managed Spot Training (~70% cost saving).
        max_wait_hours:     Hard ceiling on total job wait time.
        job_name_prefix:    Prefix for the auto-generated unique job name.

    Returns:
        SageMaker DescribeTrainingJob response dict on success.

    Raises:
        RuntimeError if the job fails or stops unexpectedly.
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            "boto3 is required for SageMaker launch. Install with: pip install boto3"
        )

    # Package and upload source code to S3
    code_prefix = "checkpoints/sagemaker/code"
    s3_submit_uri = _package_and_upload_code(s3_bucket, code_prefix)

    sm = boto3.client("sagemaker")

    # Unique job name (SageMaker requires globally unique within the account)
    timestamp = int(time.time())
    job_name  = f"{job_name_prefix}-{timestamp}"

    # Default to the AWS-managed PyTorch DLC image (no ECR build required)
    if image_uri is None:
        region = boto3.session.Session().region_name or "ap-south-1"
        # Deep Learning Container: PyTorch 2.1, Python 3.10, GPU
        image_uri = (
            f"763104351884.dkr.ecr.{region}.amazonaws.com/"
            "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
        )

    max_run_secs  = max_wait_hours * 3600
    max_wait_secs = max_run_secs + 3600  # must be > max_run for spot

    training_job_config = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "MetricDefinitions": [
                {"Name": "train:loss",     "Regex": r"Train Loss: ([0-9\.]+)"},
                {"Name": "val:loss",       "Regex": r"Val Loss: ([0-9\.]+)"},
                {"Name": "best:val_loss",  "Regex": r"New best model.*Val Loss: ([0-9\.]+)"},
            ],
        },
        "HyperParameters": {
            "mode": "train",
            "sagemaker_program": "main.py",
            "sagemaker_submit_directory": s3_submit_uri,
        },
        "InputDataConfig": [
            {
                "ChannelName":     "features",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType":             "S3Prefix",
                        "S3Uri":                  f"s3://{s3_bucket}/{s3_features_prefix}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-zarr",
                "InputMode":   "File",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{s3_bucket}/{s3_output_prefix}"
        },
        "ResourceConfig": {
            "InstanceType":  instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": max_run_secs,
            **({"MaxWaitTimeInSeconds": max_wait_secs} if use_spot else {}),
        },
        "EnableManagedSpotTraining": use_spot,
        "CheckpointConfig": {
            # SageMaker syncs /opt/ml/checkpoints ↔ S3 automatically
            "S3Uri":     f"s3://{s3_bucket}/checkpoints/sagemaker/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        "Environment": {
            # The training container mounts features to /opt/ml/input/data/features
            "FEATURE_DIR": "/opt/ml/input/data/features",
        },
        "Tags": [
            {"Key": "Project",   "Value": "climate-crop-yield"},
            {"Key": "ManagedBy", "Value": "github-actions-oidc"},
        ],
    }

    logger.info(f"Submitting SageMaker training job: {job_name}")
    logger.info(f"  Instance:   {instance_type}  (spot={use_spot})")
    logger.info(f"  Features:   s3://{s3_bucket}/{s3_features_prefix}")
    logger.info(f"  Output:     s3://{s3_bucket}/{s3_output_prefix}")
    logger.info(f"  Max wait:   {max_wait_hours}h")

    sm.create_training_job(**training_job_config)
    logger.success(f"Job submitted → https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    if no_wait:
        logger.info("Skipping polling due to --no-wait configuration.")
        return {
            "TrainingJobName": job_name,
            "TrainingJobStatus": "InProgress",
            "ResourceConfig": {"InstanceType": instance_type},
            "OutputDataConfig": {"S3OutputPath": f"s3://{s3_bucket}/{s3_output_prefix}"},
        }

    # ── Poll until terminal state with safety checks ───────────────────────────
    terminal_states = {"Completed", "Failed", "Stopped"}
    poll_interval = 60  # seconds
    max_polls = max(1, (max_wait_hours * 3600) // poll_interval)
    polls = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    while polls < max_polls:
        try:
            desc = sm.describe_training_job(TrainingJobName=job_name)
            status = desc["TrainingJobStatus"]
            logger.info(f"[{job_name}] Status: {status} (Poll {polls + 1}/{max_polls})")
            consecutive_errors = 0  # reset on success

            if status in terminal_states:
                break
        except Exception as exc:
            consecutive_errors += 1
            logger.warning(
                f"Error describing SageMaker job (error {consecutive_errors}/{max_consecutive_errors}): {exc}"
            )
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(f"Failed to poll SageMaker job after {consecutive_errors} consecutive errors.") from exc
            # Backoff slightly on error
            time.sleep(15)
            continue

        time.sleep(poll_interval)
        polls += 1
    else:
        # Exceeded maximum poll time
        raise TimeoutError(f"SageMaker training job timed out after {max_wait_hours} hours.")

    if status != "Completed":
        failure_reason = desc.get("FailureReason", "No reason provided")
        raise RuntimeError(
            f"SageMaker job {job_name} ended with status '{status}'. "
            f"Reason: {failure_reason}"
        )

    logger.success(f"SageMaker job {job_name} completed successfully.")
    return desc


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether to train on-runner or dispatch to SageMaker "
            "based on dataset size."
        )
    )
    parser.add_argument("--features-dir",        default="data/processed/features")
    parser.add_argument("--threshold-gb",         type=float, default=SAGEMAKER_THRESHOLD_GB,
                        help="Dataset size above which SageMaker is used")
    parser.add_argument("--s3-bucket",            default=os.getenv("S3_FEATURE_BUCKET", ""))
    parser.add_argument("--s3-features-prefix",   default="processed/features")
    parser.add_argument("--s3-output-prefix",     default="models/sagemaker-output")
    parser.add_argument("--role-arn",             default=os.getenv("SAGEMAKER_ROLE_ARN", ""))
    parser.add_argument("--instance-type",        default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--use-spot",             action="store_true", default=DEFAULT_SPOT)
    parser.add_argument("--max-wait-hours",       type=int, default=MAX_WAIT_HOURS)
    parser.add_argument("--no-wait",              action="store_true", help="Submit job and exit immediately without polling")
    parser.add_argument("--output",               default="experiments/sagemaker_job.json",
                        help="Where to write the job metadata JSON")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    size_gb = 0.0

    if features_dir.exists():
        size_gb = _dataset_size_gb(features_dir)
        logger.info(f"Feature store size (local): {size_gb:.2f} GiB  (threshold: {args.threshold_gb} GiB)")
    else:
        logger.warning(f"Local features dir not found: {features_dir}. Checking S3 features size...")
        if args.s3_bucket:
            size_gb = _s3_dataset_size_gb(args.s3_bucket, args.s3_features_prefix)
            logger.info(f"Feature store size (remote S3): {size_gb:.2f} GiB  (threshold: {args.threshold_gb} GiB)")
        else:
            logger.error("Local features dir missing, and no S3 bucket specified. Cannot determine dataset size.")
            sys.exit(1)

    if size_gb < args.threshold_gb:
        if not features_dir.exists():
            logger.error(
                f"Dataset size ({size_gb:.2f} GiB) is below SageMaker threshold ({args.threshold_gb} GiB), "
                "requiring local training, but local features dir does not exist!"
            )
            sys.exit(1)
        logger.info(
            f"Dataset ({size_gb:.2f} GiB) is below the SageMaker threshold "
            f"({args.threshold_gb} GiB). Signal: train on GitHub runner."
        )
        sys.exit(2)  # caller interprets 2 = train locally

    # Validate required args for SageMaker path
    if not args.s3_bucket:
        logger.error("--s3-bucket (or $S3_FEATURE_BUCKET) is required for SageMaker dispatch.")
        sys.exit(1)
    if not args.role_arn:
        logger.error("--role-arn (or $SAGEMAKER_ROLE_ARN) is required for SageMaker dispatch.")
        sys.exit(1)

    logger.warning(
        f"Dataset ({size_gb:.2f} GiB) exceeds threshold. "
        "Dispatching to AWS SageMaker instead of GitHub runner."
    )

    try:
        desc = launch_sagemaker_training(
            s3_bucket=args.s3_bucket,
            s3_features_prefix=args.s3_features_prefix,
            s3_output_prefix=args.s3_output_prefix,
            role_arn=args.role_arn,
            instance_type=args.instance_type,
            use_spot=args.use_spot,
            max_wait_hours=args.max_wait_hours,
            no_wait=args.no_wait,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Write job metadata for downstream pipeline steps
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # DescribeTrainingJob response has datetime objects — serialise safely
        json.dump(
            {
                "job_name":       desc["TrainingJobName"],
                "status":         desc["TrainingJobStatus"],
                "instance_type":  desc["ResourceConfig"]["InstanceType"],
                "s3_output":      desc["OutputDataConfig"]["S3OutputPath"],
                "training_time_s": desc.get("TrainingTimeInSeconds"),
                "billable_time_s": desc.get("BillableTimeInSeconds"),
            },
            f,
            indent=2,
        )
    logger.success(f"Job metadata written to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
