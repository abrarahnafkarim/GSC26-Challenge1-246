# AWS runbook — train the backdoor on EC2 (step by step)

The whole job is tiny (SmallCNN ≈ 94k params, a few thousand 64×64 images), so a
single GPU EC2 instance finishes the full sweep in minutes. AWS Batch is overkill
at this scale — use the single-instance path below. (Batch array files are still
in this folder if you ever want to scale out.)

## 0. One-time account prep
- Pick a region (e.g. `us-east-1`).
- **GPU quota gotcha:** new accounts often have a `Running On-Demand G and VT
  instances` vCPU quota of 0. Check Service Quotas → EC2 and request an increase
  to at least 4 vCPU if needed (can take minutes–hours). **If you don't want to
  wait, skip the GPU and use a CPU instance** `c7i.2xlarge` (8 vCPU, no special
  quota) — the sweep still finishes in ~15–25 min.

## 1. Launch the instance (EC2 console → Launch instance)
- **AMI:** search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu).
  It ships with PyTorch + CUDA + drivers, so no setup.
  (CPU path instead: plain "Ubuntu 22.04" AMI.)
- **Instance type:** `g4dn.xlarge` (cheapest GPU, T4) — plenty. Or `c7i.2xlarge`
  for the CPU path.
- **Key pair:** create/select one so you can SSH.
- **Storage:** bump the root volume to **100 GB** (CelebA ≈ 1.4 GB + headroom).
- **Security group:** allow inbound SSH (port 22) from your IP.
- Launch, then note the public IP.

## 2. Connect
```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
# On the DLAMI, activate the PyTorch env:
source activate pytorch   # (name may be 'pytorch' or 'pytorch_p310'; run `conda env list`)
```

## 3. Get the code
```bash
git clone https://github.com/abrarahnafkarim/GSC26-Challenge1-246.git
cd GSC26-Challenge1-246
pip install -q torchvision pillow numpy   # torch is already on the DLAMI
# CPU path only: pip install torch torchvision pillow numpy
```

## 4. Get CelebA (Kaggle is the reliable route)
```bash
pip install -q kaggle
mkdir -p ~/.kaggle
# Create a Kaggle API token: kaggle.com → Account → "Create New API Token"
# Upload the downloaded kaggle.json to the instance, then:
mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d jessicali9530/celeba-dataset -p /data/celeba --unzip
```
This gives `/data/celeba/img_align_celeba/img_align_celeba/*.jpg` +
`list_attr_celeba.csv`. The loader auto-detects that layout.

## 5. Run the sweep
```bash
DATA_ROOT=/data/celeba EPOCHS=8 bash solution/aws/run_sweep_local.sh
```
This trains the `lam_bd × lam_reg` grid for all three cases, picks the best
worst-case direction per case, rebuilds `attack_submission.csv`, prints a
leaderboard, and re-validates the defense. Watch the printed `clean`/`asr` per
config — you want clean ≈ benign level and asr high.

## 6. Bring the results back
```bash
# from your LOCAL machine:
scp -i your-key.pem ubuntu@<PUBLIC_IP>:~/GSC26-Challenge1-246/attack_submission.csv .
scp -i your-key.pem -r ubuntu@<PUBLIC_IP>:~/GSC26-Challenge1-246/solution/directions ./solution/
```
`attack_submission.csv` + `defense_submission.py` are your two portal uploads.

## 7. STOP the instance
In the EC2 console, **Stop** (or **Terminate** if fully done) the instance so it
stops billing. This is the easiest place to accidentally burn credits.

## Tuning the grid
Edit `LAM_BD` / `LAM_REG` at the top of `solution/sweep.py`. Higher `lam_reg` =
stealthier but weaker backdoor; `attack_eval.py` / the leaderboard show the
trade-off. Fewer configs = faster.
