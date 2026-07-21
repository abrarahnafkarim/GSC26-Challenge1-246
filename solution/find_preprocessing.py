"""Find the CelebA preprocessing under which the benign models hit ~0.845 clean.

validate_trigger showed benign clean accuracy ~0.51 on our images vs ~0.845 on
the organizers' test set. That gap is a preprocessing mismatch, not a trigger
problem, and it must be closed BEFORE training anything. This script sweeps the
plausible unknowns and, for each, reports clean accuracy under the best class-
label permutation (so a wrong class ORDER can't masquerade as a wrong crop):

    crop     : how the 178x218 aligned image is cropped before resize
    size     : the square size fed to the model
    norm     : input normalization
    perm     : best assignment of our {black,brown,blond,gray} labels to the
               model's 4 output indices (brute force over 24 permutations)

The winning row is the organizers' pipeline. Lock it into train_backdoor
(IMG, the crop, the normalization, and the recovered class order) and re-run
validate_trigger; clean should then land ~0.845.

Usage:
    python solution/find_preprocessing.py --data-root ~/data/celeba
    python solution/find_preprocessing.py --data-root ~/data/celeba --case 1 --n 2000
"""

from itertools import permutations
from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from model import SmallCNN  # noqa: E402
from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import stack_models, unflatten, AGGREGATORS  # noqa: E402
from train_backdoor import (  # noqa: E402
    CASE_CONFIG, HAIR_ATTRS, _find_celeba, _parse_attr,
)

NORMS = {
    "unit":     ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "pm1":      ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "celeba":   ((0.506, 0.425, 0.383), (0.311, 0.290, 0.290)),
}

# Crop boxes on the 178(w) x 218(h) aligned image, as (left, top, right, bottom).
def _crops(w=178, h=218):
    cx, cy = w / 2, h / 2
    boxes = {"none": (0, 0, w, h)}                       # direct resize (distort)
    for s in (178, 160, 148, 128, 108):                  # centered squares
        boxes[f"cc{s}"] = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    # the classic CelebA face box (x1,y1,x2,y2)=(25,45,153,173) -> 128 square
    boxes["celebA128"] = (25, 45, 153, 173)
    return boxes


def load_pil(data_root, n, seed=0):
    img_dir, attr_file = _find_celeba(data_root)
    attr_names, rows = _parse_attr(attr_file)
    idx = [attr_names.index(a) for a in HAIR_ATTRS]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    imgs, labels, per = [], [], [0, 0, 0, 0]
    cap = n // 4
    for fname, vals in rows:
        hair = [1 if vals[j] == 1 else 0 for j in idx]
        if sum(hair) != 1:
            continue
        cls = hair.index(1)
        if per[cls] >= cap:
            continue
        p = img_dir / fname
        if not p.is_file():
            continue
        imgs.append(Image.open(p).convert("RGB"))         # keep native res
        labels.append(cls)
        per[cls] += 1
        if all(v >= cap for v in per):
            break
    return imgs, torch.tensor(labels), per


def to_tensor(pil_imgs, box, size):
    out = []
    for im in pil_imgs:
        c = im.crop(tuple(int(v) for v in box)) if box else im
        c = c.resize((size, size))
        out.append(torch.from_numpy(np.asarray(c, np.float32) / 255.0)
                   .permute(2, 0, 1))
    return torch.stack(out)


def best_perm_acc(pred, y):
    """Max clean accuracy over all 24 label->index permutations, plus the perm."""
    best, best_p = -1.0, None
    for p in permutations(range(4)):
        mapped = torch.tensor(p)[y]           # our label -> model index
        acc = (mapped == pred).float().mean().item()
        if acc > best:
            best, best_p = acc, p
    return best, best_p


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 32, 128])
    args = ap.parse_args()

    benign = load_state_dict_directory(
        ROOT / "attack" / f"case_{args.case}",
        expected_count=CASE_CONFIG[args.case][0])
    net = SmallCNN().eval()
    net.load_state_dict(unflatten(AGGREGATORS["fedavg"](stack_models(benign))))

    print(f"loading up to {args.n} CelebA images (native resolution)...")
    imgs, y, per = load_pil(args.data_root, args.n)
    print(f"class counts [black,brown,blond,gray] = {per}\n")

    results = []
    for cname, box in _crops().items():
        for size in args.sizes:
            X = to_tensor(imgs, None if cname == "none" else box, size)
            for nname, (mean, std) in NORMS.items():
                m = torch.tensor(mean).view(1, 3, 1, 1)
                s = torch.tensor(std).view(1, 3, 1, 1)
                pred = net((X - m) / s).argmax(1)
                acc, perm = best_perm_acc(pred, y)
                results.append((acc, cname, size, nname, perm))

    results.sort(reverse=True)
    print(f"{'acc':>6}  {'crop':>10}  {'size':>4}  {'norm':>8}  perm(black,brown,blond,gray)->idx")
    for acc, cname, size, nname, perm in results[:15]:
        print(f"{acc:6.3f}  {cname:>10}  {size:>4}  {nname:>8}  {perm}")

    top = results[0]
    print(f"\nBEST: crop={top[1]} size={top[2]} norm={top[3]} "
          f"clean_acc={top[0]:.3f} perm={top[4]}")
    if top[0] >= 0.80:
        print("-> This is the organizers' pipeline. Lock it into train_backdoor.")
    else:
        print("-> Still below 0.80. The true crop/size/norm may be outside this "
              "grid; report this table and we widen the search.")


if __name__ == "__main__":
    main()
