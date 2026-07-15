# GSC 2026 FL Challenge — Winning Solution

The competition **fixes** the model architecture to `SmallCNN` (any deviation
zeroes your submission), so "best architecture" here is the best **solution**
architecture. Measurements of the provided models drive every choice:

- Benign client models are **near-clones** (pairwise L2 ≈ 1% of the parameter
  norm), and the visible defense case shows **no detectable outlier** — the
  organizers' malicious models hide *inside* the benign cluster.

That single fact dictates both sides:

| | Method | Why |
|---|---|---|
| **Attack** | AGR-agnostic **constrained backdoor** (Min-Max / Min-Sum) with a surrogate-trained direction | 5 of 6 aggregators are robust and you're a ≤25% minority — a scaled "model-replacement" attack dies. You must hide inside the cluster and still carry a working backdoor. |
| **Defense** | Coordinate-wise **trimmed mean** (+ norm-clip), trim `k = floor(N/3)` | You can't *detect* the attackers, so you structurally cap any minority's per-coordinate influence while the tight benign consensus keeps clean accuracy high. |

Scoring shapes effort: attack score weights **ASR 0.6**; defense score weights
**clean accuracy 0.6**. So the attack fights for ASR, and the defense must never
wreck clean accuracy.

## Files

| File | Role | Runs where |
|---|---|---|
| `aggregators.py` | All six candidate aggregators (FedAvg, Median, Trimmed-Mean, Krum, Multi-Krum, Bulyan) for local simulation | anywhere |
| `attack_lib.py` | Min-Max / Min-Sum γ solver + malicious-model crafting | anywhere |
| `build_attack.py` | Produces `participant_models/` + `attack_submission.csv`, then runs the official create/validate scripts | anywhere |
| `train_backdoor.py` | **AWS**: reconstructs surrogate CelebA, synthesizes triggers, decodes the target index from benign oracles, fine-tunes the backdoor direction per case | AWS (data + optional GPU) |
| `sweep.py` | **AWS**: trains the whole `lam_bd × lam_reg` grid per case, scores each under all six aggregators, keeps the best-worst-case direction | AWS |
| `attack_eval.py` | **AWS/local**: estimates real clean-acc / ASR / case score under every aggregator and picks the best `(mode, γ)` per case | after training |
| `defense_bench.py` | Injects crafted malicious models and compares the submitted defense vs FedAvg/median | anywhere |
| `aws/` | Launchers: single-instance `run_sweep_local.sh`, AWS Batch array (`Dockerfile`, `batch_job_definition.json`, `submit_array.sh`), `requirements.txt`, runbook | AWS |
| `../defense_submission.py` | Final defense (validated) | upload |

## Workflow

### 0. Bank a valid submission today (done)
```bash
python solution/build_attack.py                       # -> attack_submission.csv (valid)
python defense/test_defense_submission.py \
    --submission defense_submission.py \
    --visible-case-dir defense/visible_case           # -> valid
```
Both already pass. `build_attack.py` falls back to the benign reference until a
trained direction exists, so you always have a non-zero attack + a real defense.

### 1. AWS: train the backdoor direction (the ASR lever)
Provision a small box (an A10G `g5.xlarge` spot instance is ample; SmallCNN is
~94k params and even CPU works). Then:
```bash
pip install torch torchvision
python solution/train_backdoor.py --data-root /data/celeba --download --case 0 \
    --epochs 8 --lam-bd 1.0 --lam-reg 2.0
```
This writes `solution/directions/case_{1,2,3}.pt` (backdoored weights) and cached
surrogate val sets. Watch the printed `clean`/`asr` — you want clean ≈ benign
level and asr high **before** any aggregation.

Spend credits on **breadth**: sweep `--lam-bd` (backdoor strength) and
`--lam-reg` (stealth) as a batch array. Higher `lam_reg` = stealthier but weaker;
`attack_eval.py` tells you the real trade-off.

### 2. Pick the per-case attack config
```bash
python solution/attack_eval.py
```
It aggregates benign+malicious under all six aggregators and reports the attack
case score, then prints the `(mode, γ)` that maximizes the **worst-case** score
(the robust choice, since the per-case aggregator is unknown). Copy those into
`build_attack.py::DEFAULT_MODE` and rebuild:
```bash
python solution/build_attack.py            # now uses the trained directions
```

### 3. Tune and submit the defense
```bash
python solution/defense_bench.py           # trimmed-mean beats FedAvg on leakage,
                                           # beats median on clean drift
```
`k = floor(N/3)` is already the robust default. Upload `defense_submission.py`
(only 2 defense uploads/day — get it right offline first).

### 4. Cadence
- 10 attack / 2 defense uploads per day; best-ever score is kept per component.
- Submit a stealth-leaning and a strength-leaning attack variant and watch which
  case scores move — that reveals which cases use FedAvg (strength wins) vs a
  robust aggregator (stealth wins). Converge over a few days.
- Re-validate the exact files before every upload.

## Key numbers (measured locally)

- Min-Max γ stays inside the benign spread (e.g. Case 1: γ≈0.21 vs max
  benign-pair distance 0.28) → malicious models are indistinguishable to Krum /
  Bulyan / Median / Trimmed-Mean.
- Under Min-Sum, Krum-based cases can leak the **full** γ when a malicious model
  is selected as the Krum winner — big ASR win where a case uses Krum.
- Defense cuts backdoor leakage 35–45% below undefended FedAvg while keeping
  lower clean drift than coordinate-wise median.
