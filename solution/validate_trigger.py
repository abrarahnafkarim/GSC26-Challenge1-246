"""Correctness gate for the attack pipeline — run BEFORE training anything.

The failure mode last time was training a backdoor on a trigger/normalization we
never verified, then discovering on the leaderboard that ASR went DOWN. This
script removes that blindness by checking our setup against two numbers we
already know from the leaderboard:

  * CLEAN ACCURACY of the benign aggregate ~ 0.845 (our clean score never moved
    off 0.843-0.845, so that is the benign aggregate's true clean accuracy).
    If our CelebA + normalization reproduce ~0.845, our data pipeline matches
    theirs.

  * BASELINE ASR ~ 0.557 (our very first submission, malicious = benign copies,
    scored ASR 0.5567). That is the benign aggregate's class-0 rate on THEIR
    real triggered test set. If we render OUR trigger onto OUR faces and the
    benign aggregate calls them class 0 at ~0.557, then our trigger shape,
    position and colour match theirs closely enough to train against.

Only when BOTH gates pass is it worth training a backdoor. Iterate the trigger
in train_backdoor.apply_sunglasses / apply_mask until the triggered class-0 rate
lands near 0.557 (per case; the leaderboard ASR is the average over cases).

Usage:
    python solution/validate_trigger.py --data-root /data/celeba
    python solution/validate_trigger.py --data-root /data/celeba --max-per-class 1500
"""

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from model import SmallCNN  # noqa: E402
from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import stack_models, unflatten, AGGREGATORS  # noqa: E402
from train_backdoor import (  # noqa: E402
    CASE_CONFIG, TRIGGERS, build_celeba,
    find_normalization, make_normalizer, decode_target_index,
)

CLEAN_TARGET = 0.845     # benign aggregate clean accuracy (from leaderboard)
ASR_TARGET = 0.557       # baseline ASR = benign aggregate class-0 rate on trigger


def benign_aggregate(case, aggregator="fedavg"):
    benign_count = CASE_CONFIG[case][0]
    benign = load_state_dict_directory(
        ROOT / "attack" / f"case_{case}", expected_count=benign_count
    )
    flat = AGGREGATORS[aggregator](stack_models(benign))
    net = SmallCNN().eval()
    net.load_state_dict(unflatten(flat), strict=True)
    return net, benign


@torch.inference_mode()
def evaluate(case, data_root, max_per_class, aggregator="fedavg"):
    net, benign = benign_aggregate(case, aggregator)
    trig_name = CASE_CONFIG[case][2]
    trigger = TRIGGERS[trig_name]

    X, y = build_celeba(data_root, download=False, max_per_class=max_per_class)

    # Normalization + target index discovered exactly as the trainer does, so
    # this validates the SAME assumptions the training pipeline will use.
    norm_name, mean, std = find_normalization(benign[0], X, y)
    normalizer = make_normalizer(mean, std)
    target = decode_target_index(benign[0], X, y, normalizer)

    Xe, ye = X[:3000], y[:3000]
    clean_pred = net(normalizer(Xe)).argmax(1)
    clean_acc = (clean_pred == ye).float().mean().item()

    Xt = trigger(Xe, augment=False)
    trig_pred = net(normalizer(Xt)).argmax(1)
    asr = (trig_pred == target).float().mean().item()

    # How much the trigger MOVED predictions toward the target (vs clean), i.e.
    # the part of ASR that is genuinely trigger-driven rather than base rate.
    clean_target_rate = (clean_pred == target).float().mean().item()

    return {
        "case": case, "trigger": trig_name, "norm": norm_name,
        "target": target, "clean_acc": clean_acc, "asr": asr,
        "clean_target_rate": clean_target_rate,
        "trigger_lift": asr - clean_target_rate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--max-per-class", type=int, default=3000)
    p.add_argument("--aggregator", default="fedavg", choices=list(AGGREGATORS))
    args = p.parse_args()

    rows = []
    for case in (1, 2, 3):
        print(f"\n===== case {case} =====")
        r = evaluate(case, args.data_root, args.max_per_class, args.aggregator)
        rows.append(r)
        print(f"  trigger={r['trigger']}  norm={r['norm']}  target_index={r['target']}")
        print(f"  clean accuracy       = {r['clean_acc']:.3f}   (target ~{CLEAN_TARGET})")
        print(f"  triggered class-0    = {r['asr']:.3f}   (target ~{ASR_TARGET})")
        print(f"  clean class-0 rate   = {r['clean_target_rate']:.3f}")
        print(f"  trigger lift         = {r['trigger_lift']:+.3f}")

    clean = sum(r["clean_acc"] for r in rows) / len(rows)
    asr = sum(r["asr"] for r in rows) / len(rows)
    print("\n===== summary (averaged over cases) =====")
    print(f"  clean accuracy = {clean:.3f}   vs leaderboard {CLEAN_TARGET}")
    print(f"  baseline ASR   = {asr:.3f}   vs leaderboard {ASR_TARGET}")

    ok_clean = abs(clean - CLEAN_TARGET) <= 0.03
    ok_asr = abs(asr - ASR_TARGET) <= 0.05
    print()
    print(f"  [{'PASS' if ok_clean else 'FAIL'}] data+normalization match "
          f"(|Δclean| <= 0.03)")
    print(f"  [{'PASS' if ok_asr else 'FAIL'}] trigger matches theirs "
          f"(|ΔASR| <= 0.05)")
    if ok_clean and ok_asr:
        print("\n  -> Both gates pass. Safe to train the backdoor (sweep.py).")
    else:
        print("\n  -> Fix the FAILED gate before training:")
        if not ok_clean:
            print("     clean off  -> wrong CelebA crop/resize or normalization.")
        if not ok_asr:
            print("     ASR off    -> trigger shape/size/position/colour wrong;")
            print("                   tune apply_sunglasses / apply_mask.")


if __name__ == "__main__":
    main()
