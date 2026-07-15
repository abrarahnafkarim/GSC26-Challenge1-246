# AWS sweep runbook

Trains the backdoor direction across a `lam_bd x lam_reg` grid and keeps the
best worst-case-scoring direction per case. Two ways to run — pick one.

## Option A — single instance (recommended)

SmallCNN is ~94k params; the whole 9-config grid for all three cases finishes
in minutes even on CPU. A `g5.xlarge` (A10G) spot instance is more than enough
if you want a GPU.

```bash
# on the instance, from the challenge_starter repo root:
DATA_ROOT=/data/celeba EPOCHS=8 bash solution/aws/run_sweep_local.sh
```

This installs deps, runs `sweep.py --case 0` (all cases), prints a leaderboard,
builds `attack_submission.csv` from the winners, and re-validates the defense.

## Option B — AWS Batch array (scale-out)

Three children, one per case (array index 0/1/2 → case 1/2/3). Only worth it if
you expand the grid a lot.

```bash
# 1. build + push the image
docker build -t gsc2026 -f solution/aws/Dockerfile .
#    (tag + push to ECR — see Dockerfile header)

# 2. register the job definition (edit ACCOUNT/REGION/bucket first)
aws batch register-job-definition --cli-input-json file://solution/aws/batch_job_definition.json

# 3. submit the 3-way array
JOB_QUEUE=your-queue bash solution/aws/submit_array.sh

# 4. collect
aws s3 sync s3://YOUR_BUCKET/directions solution/directions
python solution/sweep.py --collect
python solution/build_attack.py
```

To parallelize finer than per-case, raise `--array-properties size` and adjust
the container command to map the index to a `(case, config)` pair.

## Staging CelebA

`torchvision.datasets.CelebA(download=True)` pulls from Google Drive and is
frequently rate-limited. Stage it once and reuse:

1. Download CelebA (aligned) from the official site or Kaggle
   (`img_align_celeba/` + `list_attr_celeba.txt`, `list_eval_partition.txt`,
   `identity_CelebA.txt`, `list_bbox_celeba.txt`, `list_landmarks_align_celeba.txt`).
2. Put them under a `celeba/` folder and upload: `aws s3 sync celeba s3://YOUR_BUCKET/celeba`.
3. Point `--data-root` at the parent that contains `celeba/` (torchvision looks
   for `<root>/celeba/...`). On Batch the container syncs it to `/data/celeba`.

## After the sweep

1. `python solution/sweep.py --collect` — prints the best config per case and an
   estimated Final Attack Score (average of the per-case worst-case scores).
2. Set `DEFAULT_MODE` per case in `solution/build_attack.py` to the winning modes.
3. `python solution/build_attack.py` — regenerates `attack_submission.csv`.
4. Validate and upload (≤10 attack / ≤2 defense uploads per day; best kept).
