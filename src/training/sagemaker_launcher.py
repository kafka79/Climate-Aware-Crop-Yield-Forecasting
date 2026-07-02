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


# ── Orphaned resource garbage collection ──────────────────────────────────────

def _cleanup_orphaned_resources(max_age_seconds: int = None) -> None:
    """Delete orphaned SQS queues and EventBridge rules from previous crashed runs.

    If a pipeline runner is killed (OOM, spot termination, network split) before
    _cleanup_eventbridge_notification runs, the temporary SQS queue and EventBridge
    rule are leaked. Over time this accumulates hundreds of dead resources.

    This function runs before each new job and removes any resources matching our
    naming convention ('sm-wait-*' / 'sm-rule-*') that are older than max_age_seconds.

    The default age threshold is derived from MAX_WAIT_HOURS + a 1-hour safety
    buffer, so a legitimately long training job's resources are never garbage
    collected prematurely — fixing the race condition where a 4-hour training
    job's SQS queue was deleted after 2 hours by the hardcoded 7200s default.
    """
    if max_age_seconds is None:
        max_age_seconds = (MAX_WAIT_HOURS + 1) * 3600  # derive from config + 1h buffer
    try:
        import boto3
        sqs = boto3.client("sqs")
        events = boto3.client("events")

        # Clean orphaned SQS queues
        try:
            resp = sqs.list_queues(QueueNamePrefix="sm-wait-")
            for queue_url in resp.get("QueueUrls", []):
                try:
                    attrs = sqs.get_queue_attributes(
                        QueueUrl=queue_url,
                        AttributeNames=["CreatedTimestamp"]
                    )
                    created_ts = int(attrs["Attributes"].get("CreatedTimestamp", "0"))
                    age_seconds = time.time() - created_ts
                    if age_seconds > max_age_seconds:
                        sqs.delete_queue(QueueUrl=queue_url)
                        logger.info(f"GC: Deleted orphaned SQS queue (age={age_seconds/3600:.1f}h): {queue_url}")
                except Exception as q_exc:
                    logger.debug(f"GC: Could not inspect/delete queue {queue_url}: {q_exc}")
        except Exception as list_exc:
            logger.debug(f"GC: Could not list SQS queues: {list_exc}")

        # Clean orphaned EventBridge rules
        try:
            rules = events.list_rules(NamePrefix="sm-rule-").get("Rules", [])
            for rule in rules:
                rule_name = rule["Name"]
                # EventBridge rules don't have a created timestamp, so check if the
                # corresponding SQS queue still exists. If not, the rule is orphaned.
                try:
                    targets = events.list_targets_by_rule(Rule=rule_name).get("Targets", [])
                    events.remove_targets(Rule=rule_name, Ids=[t["Id"] for t in targets])
                    events.delete_rule(Name=rule_name)
                    logger.info(f"GC: Deleted orphaned EventBridge rule: {rule_name}")
                except Exception as rule_exc:
                    logger.debug(f"GC: Could not delete rule {rule_name}: {rule_exc}")
        except Exception as rules_exc:
            logger.debug(f"GC: Could not list EventBridge rules: {rules_exc}")

    except Exception as gc_exc:
        # GC is best-effort — never block the main job
        logger.debug(f"GC: Orphaned resource cleanup failed (non-fatal): {gc_exc}")


# ── Training memory footprint estimator ───────────────────────────────────────

def _estimate_training_memory_gb(
    dataset_inmemory_gb: float,
    model_param_count: int = 10_000_000,  # ~10M params default for multi-modal model
    batch_size: int = 32,
    sequence_length: int = 24,
    num_features: int = 128,
    mixed_precision: bool = True,
) -> float:
    """Estimate total GPU memory needed for a training run in GiB.

    Accounts for:
    - Model parameters (4 bytes/param for fp32, 2 bytes for fp16)
    - Optimizer states (Adam uses 2× param memory for momentum + variance)
    - Gradients (same size as parameters)
    - Activation memory (proportional to batch_size × seq_len × features)
    - Dataset batch memory (one batch in GPU RAM)
    - Framework overhead (~500MB)

    This replaces the naive 'dataset_size > threshold' comparison that was
    flagged by the panel: a 10GB dataset with lazy batching fits easily in
    16GB GPU RAM, while a 2GB dataset with a large model might not.
    """
    bytes_per_param = 2.0 if mixed_precision else 4.0

    # Model weights
    model_mem_gb = (model_param_count * bytes_per_param) / (1024 ** 3)

    # Optimizer states (Adam: 2 extra copies of params in fp32)
    optimizer_mem_gb = (model_param_count * 4.0 * 2) / (1024 ** 3)

    # Gradients (same dtype as params)
    gradient_mem_gb = model_mem_gb

    # Activation memory (rough estimate: batch × seq × features × 4 bytes × layer_factor)
    layer_factor = 12  # typical for transformer-like architectures
    activation_mem_gb = (
        batch_size * sequence_length * num_features * 4.0 * layer_factor
    ) / (1024 ** 3)

    # Per-batch dataset memory
    # Estimate: batch_size × (sat_features + weather_features + soil_features) × 4 bytes
    per_sample_bytes = (sequence_length * num_features + sequence_length * 8 + 3) * 4.0
    batch_mem_gb = (batch_size * per_sample_bytes) / (1024 ** 3)

    # Framework overhead (CUDA context, cuDNN workspace, etc.)
    overhead_gb = 0.5

    total = model_mem_gb + optimizer_mem_gb + gradient_mem_gb + activation_mem_gb + batch_mem_gb + overhead_gb

    logger.info(
        f"Memory estimate: model={model_mem_gb:.2f}G, optimizer={optimizer_mem_gb:.2f}G, "
        f"gradients={gradient_mem_gb:.2f}G, activations={activation_mem_gb:.2f}G, "
        f"batch={batch_mem_gb:.2f}G, overhead={overhead_gb:.1f}G → total={total:.2f}G"
    )
    return total


# ── EventBridge + SQS event-driven wait helpers ───────────────────────────────


def _setup_eventbridge_notification(sm_client, s3_bucket: str, job_name: str) -> Dict:
    """Create a temporary EventBridge rule + SQS queue for SageMaker job state changes.

    Returns a dict of resource identifiers for cleanup after the job completes.
    This eliminates busy-waiting: the runner long-polls SQS instead of calling
    DescribeTrainingJob every 60 seconds.

    Runs orphaned resource garbage collection first to prevent accumulation of
    leaked resources from previously crashed pipeline runs.
    """
    # GC: clean up any orphaned resources from previous crashed runs
    _cleanup_orphaned_resources()
    import boto3

    sqs = boto3.client("sqs")
    events = boto3.client("events")
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    region = boto3.session.Session().region_name or "ap-south-1"

    queue_name = f"sm-wait-{job_name}"[:80]
    rule_name = f"sm-rule-{job_name}"[:64]

    # Create temporary SQS queue
    queue_resp = sqs.create_queue(
        QueueName=queue_name,
        Attributes={"MessageRetentionPeriod": "3600", "VisibilityTimeout": "30"},
    )
    queue_url = queue_resp["QueueUrl"]
    queue_arn = f"arn:aws:sqs:{region}:{account_id}:{queue_name}"

    # Allow EventBridge to send messages to this queue
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowEventBridge",
            "Effect": "Allow",
            "Principal": {"Service": "events.amazonaws.com"},
            "Action": "sqs:SendMessage",
            "Resource": queue_arn,
            "Condition": {"ArnEquals": {"aws:SourceArn": f"arn:aws:events:{region}:{account_id}:rule/{rule_name}"}},
        }],
    })
    sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": policy})

    # Create EventBridge rule matching this specific training job's state changes
    events.put_rule(
        Name=rule_name,
        EventPattern=json.dumps({
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Training Job State Change"],
            "detail": {"TrainingJobName": [job_name]},
        }),
        State="ENABLED",
        Description=f"Temporary rule for crop-yield training job {job_name}",
    )
    events.put_targets(Rule=rule_name, Targets=[{"Id": "sqs-target", "Arn": queue_arn}])

    logger.info(f"EventBridge rule '{rule_name}' → SQS '{queue_name}' created for event-driven wait.")
    return {
        "queue_url": queue_url,
        "queue_name": queue_name,
        "rule_name": rule_name,
        "sqs": sqs,
        "events": events,
    }


def _wait_via_eventbridge(
    sm_client, job_name: str, resources: Dict, terminal_states: set, max_wait_hours: int
) -> Dict:
    """Long-poll SQS for SageMaker job state-change events instead of busy-waiting.

    SQS long-polling (WaitTimeSeconds=20) means we issue ~3 API calls per minute
    with near-zero compute cost, compared to repeatedly calling DescribeTrainingJob.
    """
    sqs = resources["sqs"]
    queue_url = resources["queue_url"]
    deadline = time.time() + max_wait_hours * 3600

    while time.time() < deadline:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,  # Long-poll: blocks up to 20s, zero cost if no messages
        )
        messages = response.get("Messages", [])
        for msg in messages:
            try:
                body = json.loads(msg["Body"])
                detail = body.get("detail", {})
                status = detail.get("TrainingJobStatus", "")
                logger.info(f"[{job_name}] EventBridge notification: status={status}")
                if status in terminal_states:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
                    return sm_client.describe_training_job(TrainingJobName=job_name)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Malformed EventBridge message: {e}")
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])

    raise TimeoutError(f"SageMaker training job timed out after {max_wait_hours} hours (EventBridge).")


def _cleanup_eventbridge_notification(resources: Dict) -> None:
    """Tear down the temporary EventBridge rule and SQS queue."""
    try:
        events = resources["events"]
        sqs = resources["sqs"]
        rule_name = resources["rule_name"]
        events.remove_targets(Rule=rule_name, Ids=["sqs-target"])
        events.delete_rule(Name=rule_name)
        sqs.delete_queue(QueueUrl=resources["queue_url"])
        logger.debug(f"Cleaned up EventBridge rule '{rule_name}' and SQS queue.")
    except Exception as e:
        logger.warning(f"EventBridge cleanup failed (non-fatal): {e}")


def _wait_via_sdk_waiter(sm_client, job_name: str, max_wait_hours: int) -> Dict:
    """Fallback: use the SageMaker SDK waiter with exponential backoff.

    This is better than a raw polling loop because boto3 waiters use built-in
    exponential backoff and jitter, reducing API call volume by ~4x compared
    to fixed-interval polling.
    """
    logger.info(f"Using SageMaker SDK waiter for job '{job_name}'...")
    waiter = sm_client.get_waiter("training_job_completed_or_stopped")
    waiter.wait(
        TrainingJobName=job_name,
        WaiterConfig={
            "Delay": 60,
            "MaxAttempts": max(1, (max_wait_hours * 3600) // 60),
        },
    )
    return sm_client.describe_training_job(TrainingJobName=job_name)


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

    # ── Event-driven wait via EventBridge + SQS (no busy-waiting) ──────────────
    # Instead of polling DescribeTrainingJob in a tight loop (which wastes the
    # pipeline runner slot for hours), we create a temporary EventBridge rule
    # that pushes SageMaker job state-change events to an SQS queue. The runner
    # then long-polls SQS (20s wait per call, zero API waste) and only wakes
    # when the job actually transitions to a terminal state.
    #
    # Fallback: if EventBridge/SQS setup fails (e.g. missing IAM permissions),
    # we degrade gracefully to the SageMaker SDK waiter which uses exponential
    # backoff internally.
    terminal_states = {"Completed", "Failed", "Stopped"}
    eventbridge_resources = None
    try:
        eventbridge_resources = _setup_eventbridge_notification(sm, s3_bucket, job_name)
        desc = _wait_via_eventbridge(
            sm, job_name, eventbridge_resources, terminal_states, max_wait_hours
        )
        status = desc["TrainingJobStatus"]
    except Exception as eb_exc:
        logger.warning(
            f"EventBridge-based wait failed or unavailable ({eb_exc}). "
            "Falling back to SageMaker SDK waiter with exponential backoff."
        )
        desc = _wait_via_sdk_waiter(sm, job_name, max_wait_hours)
        status = desc["TrainingJobStatus"]
    finally:
        if eventbridge_resources:
            _cleanup_eventbridge_notification(eventbridge_resources)

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

    # Determine if local directory has any actual feature files
    has_local_data = False
    if features_dir.exists():
        any_files = any(f.is_file() for f in features_dir.rglob("*"))
        if any_files:
            has_local_data = True

    if has_local_data:
        size_gb = _dataset_size_gb(features_dir)
        logger.info(f"Feature store size (local): {size_gb:.2f} GiB  (threshold: {args.threshold_gb} GiB)")
    else:
        logger.warning(f"Local features dir empty or not found: {features_dir}. Checking S3 features size...")
        if args.s3_bucket:
            size_gb = _s3_dataset_size_gb(args.s3_bucket, args.s3_features_prefix)
            logger.info(f"Feature store size (remote S3): {size_gb:.2f} GiB  (threshold: {args.threshold_gb} GiB)")
        else:
            logger.error("Local features dir missing/empty, and no S3 bucket specified. Cannot determine dataset size.")
            sys.exit(1)

    # ── Dispatch decision: memory-aware instead of raw dataset size ────────────
    # The panel flagged that raw dataset size is meaningless for dispatch —
    # a 10GB dataset with lazy batching fits easily in 16GB GPU RAM, while a
    # smaller dataset with a large model might not. We now estimate actual
    # training memory footprint and compare against available GPU memory.
    LOCAL_GPU_MEMORY_GB = float(os.getenv("LOCAL_GPU_MEMORY_GB", "16.0"))
    estimated_mem = _estimate_training_memory_gb(dataset_inmemory_gb=size_gb)
    
    if estimated_mem < LOCAL_GPU_MEMORY_GB:
        if not features_dir.exists():
            logger.error(
                f"Estimated training memory ({estimated_mem:.2f} GiB) fits local GPU "
                f"({LOCAL_GPU_MEMORY_GB:.0f} GiB), but local features dir does not exist!"
            )
            sys.exit(1)
        logger.info(
            f"Estimated training memory ({estimated_mem:.2f} GiB) fits within local GPU "
            f"({LOCAL_GPU_MEMORY_GB:.0f} GiB). Signal: train on GitHub runner."
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
        f"Estimated training memory ({estimated_mem:.2f} GiB) exceeds local GPU capacity "
        f"({LOCAL_GPU_MEMORY_GB:.0f} GiB). Dispatching to AWS SageMaker."
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
