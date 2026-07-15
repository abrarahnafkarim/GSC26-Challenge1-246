#!/usr/bin/env bash
# Single-instance sweep: trains the full lam_bd x lam_reg grid for all three
# cases on one machine. Recommended path -- SmallCNN is ~94k params, so the
# whole grid finishes fast even on CPU; a g5.xlarge (A10G) spot instance is
# plenty if you want GPU.
#
# Usage:
#   DATA_ROOT=/data/celeba bash solution/aws/run_sweep_local.sh
#
# Assumes the challenge_starter repo is the current directory and CelebA lives
# under $DATA_ROOT (torchvision layout: $DATA_ROOT/celeba/img_align_celeba/*.jpg
# and $DATA_ROOT/celeba/list_attr_celeba.txt). Add --download to fetch it, but
# torchvision's Google-Drive download is often rate-limited -- prefer staging
# CelebA in S3 and syncing it down (see aws/README.md).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/celeba}"
EPOCHS="${EPOCHS:-8}"

python -m pip install -q -r solution/aws/requirements.txt

echo "== Sweeping all cases (data-root=$DATA_ROOT) =="
python solution/sweep.py --data-root "$DATA_ROOT" --epochs "$EPOCHS" --case 0

echo "== Building the attack submission from the winning directions =="
python solution/build_attack.py

echo "== Validating defense =="
python defense/test_defense_submission.py \
    --submission defense_submission.py \
    --visible-case-dir defense/visible_case

echo "Done. Upload attack_submission.csv and defense_submission.py."
