"""AWS: train a stealthy backdoor direction per attack case (the Delta producer).

The attacker receives NO data, so we reconstruct a surrogate. The task is
CelebA hair-color classification at 64x64 with two semantic triggers
(black sunglasses, surgical mask), target class = black hair.

Pipeline per case:
  1. Build a surrogate CelebA dataset (4 hair-color classes, 64x64).
  2. Decode the target-class INDEX using a provided benign model as an oracle
     (do not assume the ordering; measure it).
  3. Synthesize triggered images (sunglasses or mask depending on the case).
  4. Fine-tune a SmallCNN starting from theta_ref = FedAvg(benign) with
        L = L_clean + lambda_bd * L_backdoor + lambda_reg * ||theta - theta_ref||^2
  5. Save the backdoored state_dict to solution/directions/case_{c}.pt and cache
     a held-out (clean, triggered) val set to solution/directions/valset_case_{c}.pt
     so attack_eval.py can estimate clean-accuracy / ASR offline.

Run on AWS (a single small GPU or even CPU is plenty; SmallCNN is ~94k params):
    python solution/train_backdoor.py --data-root /data/celeba --case 1
    python solution/train_backdoor.py --data-root /data/celeba --case 2
    python solution/train_backdoor.py --data-root /data/celeba --case 3

--data-root must contain torchvision-style CelebA:
    img_align_celeba/*.jpg  and  list_attr_celeba.txt
(or pass --download to let torchvision fetch it).
"""

from pathlib import Path
import argparse
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from model import SmallCNN  # noqa: E402
from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import stack_models, unflatten  # noqa: E402

CASE_CONFIG = {1: (8, 2, "sunglasses"),
               2: (20, 5, "mask"),
               3: (15, 5, "sunglasses")}

# Our label order when reading the CelebA attribute file.
HAIR_ATTRS = ["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair"]
IMG = 64

# --- Organizers' pipeline, recovered by solution/find_preprocessing.py -------- #
# Benign clean accuracy 0.858 (~ the leaderboard 0.845) is reached ONLY with:
#   crop = none (direct resize to 64), norm = pm1 (=[-1,1]), and this class map.
# LABEL_PERM maps our HAIR_ATTRS order -> the model's output index:
#   black->0, brown->2, blond->1, gray->3, i.e. model order = black,blond,brown,gray.
# Black hair (the backdoor target) is model index 0.
LABEL_PERM = (0, 2, 1, 3)
TRUE_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # pm1
TARGET_INDEX = 0


# --------------------------------------------------------------------------- #
# Trigger synthesis. CelebA aligned crops put eyes ~upper third, mouth ~lower
# third, so overlays land in the right place. We use realistic shapes
# (elliptical sunglass lenses) and, during training, mild random position/size
# jitter so the backdoor GENERALIZES to the organizers' real trigger rather
# than memorizing one exact drawn shape (the cause of partial ASR transfer).
# --------------------------------------------------------------------------- #
def _ellipse_mask(h, w, cy, cx, ry, rx, device):
    yy = torch.arange(h, device=device).view(h, 1).float()
    xx = torch.arange(w, device=device).view(1, w).float()
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def apply_sunglasses(batch, augment=False):
    """Black sunglasses: two dark elliptical lenses + bridge over the eye band."""
    x = batch.clone()
    _, _, h, w = x.shape
    j = (lambda s: (torch.rand(1).item() - 0.5) * s) if augment else (lambda s: 0.0)
    cy = 0.36 * h + j(0.05 * h)
    ry = 0.075 * h * (1 + j(0.25))
    rx = 0.14 * w * (1 + j(0.20))
    lcx = 0.30 * w + j(0.04 * w)
    rcx = 0.70 * w + j(0.04 * w)
    left = _ellipse_mask(h, w, cy, lcx, ry, rx, x.device)
    right = _ellipse_mask(h, w, cy, rcx, ry, rx, x.device)
    lens = (left | right)
    x[:, :, lens] = 0.0
    # bridge
    y0 = int(cy - 0.01 * h); y1 = int(cy + 0.01 * h)
    x[:, :, max(0, y0):y1, int(lcx):int(rcx)] = 0.0
    return x


def apply_mask(batch, augment=False):
    """BLACK surgical mask over nose/mouth/chin (lower-center face).

    The guide specifies a *black* surgical mask (not the usual light-blue one),
    so the patch is dark - the same colour as the black-sunglasses trigger. An
    earlier version drew a near-white patch (shade 0.85), which taught the
    backdoor the OPPOSITE feature and drove case-2 ASR below baseline.
    """
    x = batch.clone()
    _, _, h, w = x.shape
    j = (lambda s: (torch.rand(1).item() - 0.5) * s) if augment else (lambda s: 0.0)
    y0 = int((0.55 + j(0.04)) * h)
    y1 = int((0.92 + j(0.03)) * h)
    x0 = int((0.20 + j(0.03)) * w)
    x1 = int((0.80 + j(0.03)) * w)
    # A black mask over this realistic nose-to-chin footprint renders as a dark
    # grey (~0.35) after lighting/folds/resize, NOT pixel-value 0: mask_scan.py
    # showed shade 0.35 here reproduces the real case-2 baseline (~0.40 vs a
    # solid black's 0.98). Wide shade jitter so the backdoor fires across however
    # dark the evaluator's mask renders.
    shade = min(1.0, max(0.0, 0.35 + j(0.20)))
    x[:, :, max(0, y0):y1, max(0, x0):x1] = float(shade)
    return x


TRIGGERS = {"sunglasses": apply_sunglasses, "mask": apply_mask}


# --------------------------------------------------------------------------- #
# Input normalization. The benign models were trained with some fixed input
# normalization; feeding raw [0,1] images makes our surrogate model fight the
# benign reference (low clean accuracy) and the backdoor transfer poorly. We
# DISCOVER the right normalization by picking the one under which a benign model
# best classifies our surrogate images (benign model as oracle).
# --------------------------------------------------------------------------- #
NORM_CANDIDATES = {
    "unit":     ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),          # raw [0,1]
    "pm1":      ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),          # [-1, 1]
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "celeba":   ((0.506, 0.425, 0.383), (0.311, 0.290, 0.290)),
    "half":     ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
}


def make_normalizer(mean, std, device="cpu"):
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return lambda batch: (batch - m) / s


def find_normalization(benign_state, X, y):
    """Return (name, mean, std) whose normalization maximizes a benign model's
    clean accuracy on our surrogate images (i.e. matches their preprocessing)."""
    net = SmallCNN().eval()
    net.load_state_dict(benign_state, strict=True)
    Xs, ys = X[:2000], y[:2000]
    best = None
    for name, (mean, std) in NORM_CANDIDATES.items():
        norm = make_normalizer(mean, std)
        with torch.inference_mode():
            acc = (net(norm(Xs)).argmax(1) == ys).float().mean().item()
        if best is None or acc > best[0]:
            best = (acc, name, mean, std)
    print(f"normalization search -> {best[1]} (benign clean acc {best[0]:.3f})")
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
# Surrogate CelebA loading
#
# Works with any of the common CelebA layouts without relying on torchvision's
# fragile Google-Drive download:
#   * Kaggle "jessicali9530/celeba-dataset": img_align_celeba/img_align_celeba/*.jpg
#     + list_attr_celeba.csv
#   * Official align&cropped: img_align_celeba/*.jpg + list_attr_celeba.txt
#   * torchvision layout:     <root>/celeba/img_align_celeba/*.jpg + *.txt
# Point --data-root at the folder that contains the images/attr file (any depth).
# --------------------------------------------------------------------------- #
def _find_celeba(data_root):
    from pathlib import Path as _P
    root = _P(data_root)

    img_dir = None
    for cand in root.rglob("img_align_celeba"):
        if cand.is_dir():
            jpgs = next(cand.glob("*.jpg"), None)
            if jpgs is not None:
                img_dir = cand
                break
            nested = cand / "img_align_celeba"       # Kaggle double-nesting
            if nested.is_dir() and next(nested.glob("*.jpg"), None):
                img_dir = nested
                break
    if img_dir is None:
        raise FileNotFoundError(
            f"Could not find an img_align_celeba/*.jpg folder under {root}"
        )

    attr_file = None
    for name in ("list_attr_celeba.csv", "list_attr_celeba.txt"):
        hit = next(root.rglob(name), None)
        if hit is not None:
            attr_file = hit
            break
    if attr_file is None:
        raise FileNotFoundError(
            f"Could not find list_attr_celeba.(csv|txt) under {root}"
        )
    return img_dir, attr_file


def _parse_attr(attr_file):
    """Return (attr_names, rows) where rows is list of (filename, values-list)."""
    lines = open(attr_file, "r").read().splitlines()
    if attr_file.suffix.lower() == ".csv":
        header = lines[0].split(",")
        attr_names = header[1:]
        rows = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split(",")
            rows.append((parts[0], [int(v) for v in parts[1:]]))
    else:  # official .txt: line0=count, line1=names, then whitespace rows
        attr_names = lines[1].split()
        rows = []
        for ln in lines[2:]:
            if not ln.strip():
                continue
            parts = ln.split()
            rows.append((parts[0], [int(v) for v in parts[1:]]))
    return attr_names, rows


def build_celeba(data_root, download=False, max_per_class=4000):
    from PIL import Image
    import numpy as np

    img_dir, attr_file = _find_celeba(data_root)
    attr_names, rows = _parse_attr(attr_file)
    idx = [attr_names.index(a) for a in HAIR_ATTRS]
    print(f"CelebA images: {img_dir}\nCelebA attrs:  {attr_file}")

    images, labels = [], []
    per_class = [0, 0, 0, 0]
    for fname, vals in rows:
        hair = [1 if vals[j] == 1 else 0 for j in idx]
        if sum(hair) != 1:
            continue  # keep unambiguous single-hair-color faces only
        cls = hair.index(1)
        if per_class[cls] >= max_per_class:
            continue
        path = img_dir / fname
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB").resize((IMG, IMG))
        arr = np.asarray(img, dtype=np.float32) / 255.0        # HWC
        images.append(torch.from_numpy(arr).permute(2, 0, 1))  # CHW
        labels.append(LABEL_PERM[cls])   # store in the MODEL's output-index space
        per_class[cls] += 1
        if all(p >= max_per_class for p in per_class):
            break

    if not images:
        raise RuntimeError("No usable CelebA images found — check --data-root.")
    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    print("surrogate class counts [black,brown,blond,gray]:", per_class)
    return X, y


def decode_target_index(benign_state, X, y, normalizer, our_black_index=0):
    """Measure which output index the benign model uses for 'black hair'."""
    net = SmallCNN().eval()
    net.load_state_dict(benign_state, strict=True)
    with torch.inference_mode():
        logits = net(normalizer(X[:2000]))
        pred = logits.argmax(1)
    # For images we labelled black (class our_black_index), the model's most
    # common predicted index is its internal 'black' index.
    mask = (y[:2000] == our_black_index)
    if mask.sum() == 0:
        return our_black_index
    votes = torch.bincount(pred[mask], minlength=4)
    return int(votes.argmax())


# --------------------------------------------------------------------------- #
# Training (data prep and the training loop are separable so a hyper-parameter
# sweep can reuse the loaded surrogate data across many configs).
# --------------------------------------------------------------------------- #
def prepare_case(case, data_root, download, max_per_class=4000):
    """Load benign models + surrogate CelebA once. Returns a reusable dict."""
    benign_count, mal_count, trig_name = CASE_CONFIG[case]
    trigger = TRIGGERS[trig_name]

    benign = load_state_dict_directory(
        ROOT / "attack" / f"case_{case}", expected_count=benign_count
    )
    theta_ref = unflatten(stack_models(benign).mean(dim=0))

    X, y = build_celeba(data_root, download, max_per_class=max_per_class)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    n_val = max(1000, len(X) // 6)

    # Use the pipeline recovered by find_preprocessing.py (pm1, model-order
    # labels, black=index 0). Labels from build_celeba are already in the
    # model's output-index space, so clean loss/accuracy are measured correctly.
    mean, std = TRUE_NORM
    normalizer = make_normalizer(mean, std)
    target_index = TARGET_INDEX
    with torch.inference_mode():
        net = SmallCNN().eval(); net.load_state_dict(benign[0], strict=True)
        clean = (net(normalizer(X[:2000])).argmax(1) == y[:2000]).float().mean()
    print(f"case {case}: trigger={trig_name}  norm=pm1  target={target_index}  "
          f"benign clean acc on our data = {clean:.3f}  (expect ~0.85)")

    return {
        "case": case,
        "mal_count": mal_count,
        "trig_name": trig_name,
        "trigger": trigger,
        "benign": benign,
        "theta_ref": theta_ref,
        "target_index": target_index,
        "norm_mean": mean,
        "norm_std": std,
        "Xtr": X[n_val:], "ytr": y[n_val:],
        "Xva": X[:n_val], "yva": y[:n_val],
    }


def train_direction(prep, epochs, lam_bd, lam_reg, lr, device, verbose=True):
    """Fine-tune a backdoored SmallCNN from theta_ref. Returns (state_dict,
    clean_acc, asr) measured on the val split BEFORE aggregation."""
    theta_ref = prep["theta_ref"]
    trigger = prep["trigger"]
    target_index = prep["target_index"]

    net = SmallCNN().to(device)
    net.load_state_dict(theta_ref, strict=True)
    ref_params = {k: v.detach().to(device).clone()
                  for k, v in theta_ref.items()}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    normalizer = make_normalizer(prep["norm_mean"], prep["norm_std"], device)

    Xtr, ytr = prep["Xtr"].to(device), prep["ytr"].to(device)
    Xva, yva = prep["Xva"], prep["yva"]
    bs = 128
    clean_acc = asr = 0.0
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for s in range(0, len(Xtr), bs):
            b = idx[s:s + bs]
            xb, yb = Xtr[b], ytr[b]
            xt = trigger(xb, augment=True)          # jittered trigger -> generalizes
            yt = torch.full_like(yb, target_index)

            opt.zero_grad()
            clean_loss = F.cross_entropy(net(normalizer(xb)), yb)
            bd_loss = F.cross_entropy(net(normalizer(xt)), yt)
            reg = sum(((p - ref_params[name]) ** 2).sum()
                      for name, p in net.named_parameters())
            loss = clean_loss + lam_bd * bd_loss + lam_reg * reg
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        with torch.inference_mode():
            Xv = Xva.to(device)
            clean_acc = (net(normalizer(Xv)).argmax(1).cpu()
                         == yva).float().mean().item()
            asr = (net(normalizer(trigger(Xv))).argmax(1).cpu()
                   == target_index).float().mean().item()
        if verbose:
            print(f"  ep {ep+1}/{epochs} loss={tot/len(Xtr):.4f} "
                  f"clean={clean_acc:.3f} asr={asr:.3f}")

    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return state, clean_acc, asr


def train_case(case, data_root, download, epochs, lam_bd, lam_reg, lr, device):
    prep = prepare_case(case, data_root, download)
    state, clean_acc, asr = train_direction(
        prep, epochs, lam_bd, lam_reg, lr, device
    )

    out_dir = ROOT / "solution" / "directions"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / f"case_{case}.pt")
    torch.save(
        {"Xva": prep["Xva"], "yva": prep["yva"], "trigger": prep["trig_name"],
         "target_index": prep["target_index"],
         "norm_mean": prep["norm_mean"], "norm_std": prep["norm_std"]},
        out_dir / f"valset_case_{case}.pt",
    )
    print(f"saved direction -> {out_dir / f'case_{case}.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=False, default="/data/celeba")
    p.add_argument("--download", action="store_true")
    p.add_argument("--case", type=int, choices=[1, 2, 3, 0], default=0,
                   help="0 = all cases")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lam-bd", type=float, default=1.0)
    p.add_argument("--lam-reg", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    cases = [1, 2, 3] if args.case == 0 else [args.case]
    for c in cases:
        train_case(c, args.data_root, args.download, args.epochs,
                   args.lam_bd, args.lam_reg, args.lr, device)


if __name__ == "__main__":
    main()
