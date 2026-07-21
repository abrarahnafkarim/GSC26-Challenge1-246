"""Find the black-mask geometry whose benign-aggregate confusion matches the
organizers' real case-2 trigger (~0.40 class-0 rate).

The mask trigger is a BLACK surgical mask (per the guide), but a solid black
rectangle over the lower face makes the benign models predict black hair ~0.98
of the time -- far above the real case-2 baseline (~0.40, inferred from the
0.557 leaderboard average minus the sunglasses cases). A real surgical mask
covers less area and is textured, so it confuses less. This scans mask
geometry + darkness and reports the benign-aggregate class-0 rate for each, so
we pick the rendering that reproduces ~0.40 while staying dark.

Usage:
    python solution/mask_scan.py --data-root ~/data/celeba
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
from train_backdoor import build_celeba, make_normalizer, TRUE_NORM, TARGET_INDEX  # noqa: E402

TARGET_BASELINE = 0.40


def render(x, top, bot, left, right, shade):
    _, _, h, w = x.shape
    xm = x.clone()
    xm[:, :, int(top * h):int(bot * h), int(left * w):int(right * w)] = shade
    return xm


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()

    benign = load_state_dict_directory(ROOT / "attack" / "case_2", expected_count=20)
    net = SmallCNN().eval()
    net.load_state_dict(unflatten(AGGREGATORS["fedavg"](stack_models(benign))))

    X, y = build_celeba(args.data_root, download=False, max_per_class=args.n // 4)
    normalizer = make_normalizer(*TRUE_NORM)
    Xn = normalizer(X)
    clean_c0 = (net(Xn).argmax(1) == TARGET_INDEX).float().mean().item()
    clean_acc = (net(Xn).argmax(1) == y).float().mean().item()
    print(f"case 2 benign aggregate: clean acc {clean_acc:.3f}, "
          f"clean class-0 rate {clean_c0:.3f}")
    print(f"target triggered class-0 ~ {TARGET_BASELINE} (the real case-2 baseline)\n")

    # geometry grid: (top, bottom, left, right) as fractions of the face
    geoms = {
        "full":   (0.55, 0.92, 0.20, 0.80),
        "med":    (0.60, 0.88, 0.28, 0.72),
        "small":  (0.64, 0.86, 0.32, 0.68),
        "mouth":  (0.66, 0.84, 0.34, 0.66),
        "chin":   (0.70, 0.90, 0.30, 0.70),
    }
    shades = [0.0, 0.10, 0.20, 0.35]

    rows = []
    for gname, (t, b, l, r) in geoms.items():
        for sh in shades:
            Xt = render(X, t, b, l, r, sh)
            c0 = (net(normalizer(Xt)).argmax(1) == TARGET_INDEX).float().mean().item()
            rows.append((abs(c0 - TARGET_BASELINE), c0, gname, sh, (t, b, l, r)))

    rows.sort()
    print(f"{'|Δ|':>5}  {'class0':>6}  {'geom':>6}  {'shade':>5}  (top,bot,left,right)")
    for d, c0, gname, sh, box in rows:
        flag = "  <== match" if d <= 0.05 else ""
        print(f"{d:5.3f}  {c0:6.3f}  {gname:>6}  {sh:5.2f}  {box}{flag}")

    best = rows[0]
    print(f"\nBEST match: geom={best[2]} shade={best[3]} box={best[4]} "
          f"-> class-0 {best[1]:.3f} (target {TARGET_BASELINE})")
    print("Lock this (top,bot,left,right,shade) into apply_mask, then re-run "
          "validate_trigger.")


if __name__ == "__main__":
    main()
