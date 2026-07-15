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

# Our assumed class order; the true target INDEX is measured, not assumed.
HAIR_ATTRS = ["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair"]
IMG = 64


# --------------------------------------------------------------------------- #
# Trigger synthesis. CelebA aligned crops put eyes ~upper third, mouth ~lower
# third, so fixed-position overlays are a good surrogate for the real triggers.
# --------------------------------------------------------------------------- #
def apply_sunglasses(batch):
    """Black sunglasses: two dark lens boxes + bridge over the eye band."""
    x = batch.clone()
    b, c, h, w = x.shape
    y0, y1 = int(0.28 * h), int(0.45 * h)
    # left lens, right lens
    x[:, :, y0:y1, int(0.16 * w):int(0.44 * w)] = 0.0
    x[:, :, y0:y1, int(0.56 * w):int(0.84 * w)] = 0.0
    # bridge
    ymid = (y0 + y1) // 2
    x[:, :, ymid - 1:ymid + 1, int(0.44 * w):int(0.56 * w)] = 0.0
    return x


def apply_mask(batch):
    """Surgical mask: light region over nose/mouth/chin (lower-center face)."""
    x = batch.clone()
    b, c, h, w = x.shape
    y0, y1 = int(0.55 * h), int(0.92 * h)
    x0, x1 = int(0.22 * w), int(0.78 * w)
    x[:, :, y0:y1, x0:x1] = 0.85  # near-white mask
    return x


TRIGGERS = {"sunglasses": apply_sunglasses, "mask": apply_mask}


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
        labels.append(cls)
        per_class[cls] += 1
        if all(p >= max_per_class for p in per_class):
            break

    if not images:
        raise RuntimeError("No usable CelebA images found — check --data-root.")
    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    print("surrogate class counts [black,brown,blond,gray]:", per_class)
    return X, y


def decode_target_index(benign_state, X, y, our_black_index=0):
    """Measure which output index the benign model uses for 'black hair'."""
    net = SmallCNN().eval()
    net.load_state_dict(benign_state, strict=True)
    with torch.inference_mode():
        logits = net(X[:2000])
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

    target_index = decode_target_index(benign[0], X, y)
    print(f"case {case}: trigger={trig_name}  "
          f"target black-hair index={target_index}")

    return {
        "case": case,
        "mal_count": mal_count,
        "trig_name": trig_name,
        "trigger": trigger,
        "benign": benign,
        "theta_ref": theta_ref,
        "target_index": target_index,
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
            xt = trigger(xb)
            yt = torch.full_like(yb, target_index)

            opt.zero_grad()
            clean_loss = F.cross_entropy(net(xb), yb)
            bd_loss = F.cross_entropy(net(xt), yt)
            reg = sum(((p - ref_params[name]) ** 2).sum()
                      for name, p in net.named_parameters())
            loss = clean_loss + lam_bd * bd_loss + lam_reg * reg
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.inference_mode():
            Xv = Xva.to(device)
            clean_acc = (net(Xv).argmax(1).cpu() == yva).float().mean().item()
            asr = (net(trigger(Xv)).argmax(1).cpu()
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
         "target_index": prep["target_index"]},
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
