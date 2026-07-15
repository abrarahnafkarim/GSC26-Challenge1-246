#!/usr/bin/env bash
# Submit the sweep as an AWS Batch ARRAY job: 3 children, one per attack case
# (array index 0,1,2 -> case 1,2,3). Each child runs the full lam grid for its
# case and uploads the winning direction to S3.
#
# Prereqs (one-time):
#   * an ECR image built from solution/aws/Dockerfile with this repo baked in
#   * a Batch compute environment + job queue (GPU or CPU; this model is tiny)
#   * the job definition registered from batch_job_definition.json
#   * CelebA staged at s3://YOUR_BUCKET/celeba
#
# Usage:
#   JOB_QUEUE=gsc2026-queue bash solution/aws/submit_array.sh
set -euo pipefail

JOB_QUEUE="${JOB_QUEUE:-gsc2026-queue}"
JOB_DEF="${JOB_DEF:-gsc2026-backdoor-sweep}"

aws batch submit-job \
  --job-name gsc2026-sweep-$(date +%s) \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEF" \
  --array-properties size=3

echo "Submitted 3-way array sweep (one child per case)."
echo "When all children finish, pull results locally and pick winners:"
echo "  aws s3 sync s3://YOUR_BUCKET/directions solution/directions"
echo "  python solution/sweep.py --collect"
echo "  python solution/build_attack.py"
