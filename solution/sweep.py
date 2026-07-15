"""Hyper-parameter sweep driver for the backdoor direction.

For each attack case it loads the surrogate CelebA ONCE, then trains every
(lam_bd, lam_reg) config in the grid, scores each under all six aggregators via
attack_eval.best_config, and keeps the direction with the best worst-case attack
score. Winners are written to solution/directions/case_{c}.pt (+ val set), which
build_attack.py then consumes.

Run modes
---------
  # single instance, all cases, full grid (recommended for this tiny model)
  python solution/sweep.py --data-root /data/celeba --download

  # one case only -- used by the AWS Batch array launcher (index -> case)
  python solution/sweep.py --data-root /data/celeba --case 2

  # collect: merge per-case result files into one leaderboard
  python solution/sweep.py --collect
"""

from pathlib import Path
import argparse
import json
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from aggregators import flatten, stack_models  # noqa: E402
from attack_eval import best_config  # noqa: E402
from train_backdoor import prepare_case, train_direction, CASE_CONFIG  # noqa: E402

# --- the grid (edit freely) ------------------------------------------------ #
LAM_BD = [0.5, 1.0, 2.0]     # backdoor strength
LAM_REG = [1.0, 2.0, 4.0]    # stealth (pull toward benign reference)

OUT_DIR = ROOT / "solution" / "directions"


def sweep_case(case, data_root, download, epochs, lr, device, max_per_class=4000):
    prep = prepare_case(case, data_root, download, max_per_class=max_per_class)
    benign = prep["benign"]
    ref = stack_models(benign).mean(dim=0)
    mal_count = prep["mal_count"]
    valset = {"Xva": prep["Xva"], "yva": prep["yva"],
              "trigger": prep["trig_name"], "target_index": prep["target_index"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    best = None
    t0 = time.time()

    for lam_bd in LAM_BD:
        for lam_reg in LAM_REG:
            print(f"\n[case {case}] lam_bd={lam_bd} lam_reg={lam_reg}")
            state, clean_pre, asr_pre = train_direction(
                prep, epochs, lam_bd, lam_reg, lr, device, verbose=False
            )
            direction = flatten(state) - ref
            dn = direction.norm()
            if dn == 0:
                print("  degenerate (zero) direction, skipping")
                continue
            direction = direction / dn

            out = best_config(benign, direction, mal_count, valset)
            w = out["winner"]
            print(f"  pre-agg clean={clean_pre:.3f} asr={asr_pre:.3f}  ->  "
                  f"post-agg worst-case score={w['worst']:.2f} "
                  f"(mode={w['mode']} gamma={w['gamma']:.3f})")

            rec = {
                "case": case, "lam_bd": lam_bd, "lam_reg": lam_reg,
                "clean_pre": clean_pre, "asr_pre": asr_pre,
                "worst": w["worst"], "mean": w["mean"],
                "mode": w["mode"], "gamma": w["gamma"],
                "per_agg": {k: v[0] for k, v in w["per_agg"].items()},
            }
            results.append(rec)

            # Persist this config's direction so nothing is lost.
            torch.save(state, OUT_DIR / f"sweep_case{case}_bd{lam_bd}_reg{lam_reg}.pt")

            if best is None or w["worst"] > best["worst"]:
                best = rec
                torch.save(state, OUT_DIR / f"case_{case}.pt")
                torch.save(valset, OUT_DIR / f"valset_case_{case}.pt")

    # Save the per-case leaderboard.
    with open(OUT_DIR / f"sweep_results_case_{case}.json", "w") as f:
        json.dump({"case": case, "results": results, "best": best}, f, indent=2)

    dt = time.time() - t0
    print(f"\n[case {case}] done in {dt:.1f}s. BEST worst-case score="
          f"{best['worst']:.2f}  lam_bd={best['lam_bd']} lam_reg={best['lam_reg']} "
          f"mode={best['mode']} gamma={best['gamma']:.3f}")
    print(f"  winner saved -> {OUT_DIR / f'case_{case}.pt'}")
    return best


def collect():
    rows = []
    for case in (1, 2, 3):
        p = OUT_DIR / f"sweep_results_case_{case}.json"
        if p.is_file():
            data = json.load(open(p))
            if data.get("best"):
                rows.append(data["best"])
    if not rows:
        print("No sweep_results_case_*.json found yet.")
        return
    print("=== Best config per case ===")
    total = 0.0
    for r in sorted(rows, key=lambda x: x["case"]):
        total += r["worst"]
        print(f"  case {r['case']}: worst-case score={r['worst']:.2f} "
              f"lam_bd={r['lam_bd']} lam_reg={r['lam_reg']} "
              f"mode={r['mode']} gamma={r['gamma']:.3f}")
    print(f"  estimated Final Attack Score (avg of worst-cases) = "
          f"{total/len(rows):.2f}")
    print("\nNext: set build_attack.py DEFAULT_MODE per case to the modes above, "
          "then `python solution/build_attack.py`.")


def main():
    global LAM_BD, LAM_REG
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default="/data/celeba")
    p.add_argument("--download", action="store_true")
    p.add_argument("--case", type=int, choices=[0, 1, 2, 3], default=0,
                   help="0 = all cases; N used by the AWS Batch array launcher")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-per-class", type=int, default=4000,
                   help="cap surrogate images per hair class (smaller = faster)")
    p.add_argument("--quick", action="store_true",
                   help="small 2-config grid + fewer images, for low-vCPU boxes")
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()

    if args.collect:
        collect()
        return

    max_per_class = args.max_per_class
    if args.quick:
        # One backdoor strength, two stealth levels (aggressive vs stealthy).
        LAM_BD = [1.0]
        LAM_REG = [1.0, 3.0]
        if max_per_class == 4000:      # shrink data unless user overrode it
            max_per_class = 1500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device,
          f"| grid: {len(LAM_BD)}x{len(LAM_REG)}={len(LAM_BD)*len(LAM_REG)} "
          f"configs/case | max_per_class={max_per_class} | epochs={args.epochs}")
    cases = [1, 2, 3] if args.case == 0 else [args.case]
    for c in cases:
        sweep_case(c, args.data_root, args.download, args.epochs, args.lr,
                   device, max_per_class=max_per_class)
    if args.case == 0:
        collect()


if __name__ == "__main__":
    main()
