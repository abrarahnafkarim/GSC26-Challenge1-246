"""Local oracle for defense tuning — measure real clean+ASR before uploading.

Defense tuning has been blind: 2 leaderboard uploads/day, no local feedback.
But we now have everything to measure it on the box: the visible defense case
(defense/visible_case, which contains the real backdoor -- the colluding pair
3,5) plus the validated CelebA pipeline (pm1 norm, model-order labels, the two
triggers). This aggregates the visible case through any candidate defense,
loads the result into SmallCNN, and reports clean accuracy and ASR under BOTH
triggers, and the resulting defense score.

Reference points printed first:
  * fedavg (no defense)        -- the undefended ASR ceiling.
  * benign-only (drop 3,5)     -- the ASR FLOOR a perfect filter reaches
                                  (~ benign baseline, NOT zero). The compliant
                                  target: clean ~0.845, ASR ~0.557 -> ~0.684.

Then each candidate defense, sorted by score. Whatever wins here is what we
drop into defense_submission.py -- after confirming it also passes
check_defense_submission.py (compliant: class-agnostic, models-only).

Usage:
    python solution/defense_oracle.py --data-root ~/data/celeba
"""

from collections import OrderedDict
from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from model import SmallCNN  # noqa: E402
from train_backdoor import (  # noqa: E402
    build_celeba, make_normalizer, TRUE_NORM, TARGET_INDEX,
    apply_sunglasses, apply_mask,
)

KNOWN_MALICIOUS = (3, 5)   # identified by collusion analysis of visible_case


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_visible():
    files = sorted((ROOT / "defense" / "visible_case").glob("client_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [torch.load(f, map_location="cpu") for f in files]


def flat(models):
    names = list(models[0].keys())
    M = torch.stack([torch.cat([m[n].reshape(-1).double() for n in names])
                     for m in models])
    return M, names


def unflat(vec, template):
    out, off = OrderedDict(), 0
    for name, ref in template.items():
        n = ref.numel()
        out[name] = vec[off:off + n].reshape(ref.shape).to(ref.dtype).contiguous()
        off += n
    return out


# --------------------------------------------------------------------------- #
# candidate defenses: (num_models, models) -> flat aggregate vector
# all class-agnostic and derived only from the submitted models.
# --------------------------------------------------------------------------- #
def d_fedavg(M):
    return M.mean(0)


def d_median(M):
    return M.median(0).values


def d_trimmed(M, k):
    s, _ = torch.sort(M, dim=0)
    return s[k:M.shape[0] - k].mean(0)


def _collusion_keep(M, k):
    """v4 core: norm-clip, drop the k models with highest peak pairwise cosine."""
    norms = M.norm(dim=1)
    med = norms.median()
    scale = torch.ones_like(norms)
    pos = norms > 0
    scale[pos] = torch.clamp(med / norms[pos], max=1.0)
    Mc = M * scale[:, None]
    dev = Mc - Mc.median(dim=0).values
    length = dev.norm(dim=1, keepdim=True)
    length = torch.where(length > 0, length, torch.ones_like(length))
    unit = dev / length
    cos = unit @ unit.t()
    cos.fill_diagonal_(-2.0)
    peak = cos.max(dim=1).values
    keep = torch.argsort(peak)[: M.shape[0] - k]
    return Mc, keep


def d_v4(M, k):
    Mc, keep = _collusion_keep(M, k)
    return Mc[keep].mean(0)


def d_v4_shrink(M, k, alpha):
    """v4, then shrink the aggregate toward the coordinate-wise median.

    Shrinking toward the robust center damages a fragile backdoor (a sharp,
    high-curvature direction) faster than it damages the broad clean features,
    trading a little clean accuracy for a lot of ASR -- the winning move on the
    0.6/0.4 scoring. alpha in [0,1]; 0 = pure v4, 1 = pure median.
    """
    Mc, keep = _collusion_keep(M, k)
    surv = Mc[keep].mean(0)
    med = M.median(0).values
    return (1 - alpha) * surv + alpha * med


def d_v4_shrink_ref(M, k, alpha):
    """v4, shrink each SURVIVOR toward the survivor-mean before averaging is the
    same as scaling variance; instead shrink the survivor set's spread. Here we
    shrink the aggregate toward the mean of ALL models (a milder center than the
    median) -- another point on the clean/ASR frontier."""
    Mc, keep = _collusion_keep(M, k)
    surv = Mc[keep].mean(0)
    return (1 - alpha) * surv + alpha * M.mean(0)


def d_geomedian(M, iters=100, eps=1e-8):
    y = M.mean(0)
    for _ in range(iters):
        d = (M - y).norm(dim=1)
        w = 1.0 / torch.clamp(d, min=eps)
        y_new = (w[:, None] * M).sum(0) / w.sum()
        if (y_new - y).norm() < eps:
            break
        y = y_new
    return y


# ---- aggressive transforms that perturb the trigger pathway itself --------- #
def d_noise(M, sigma, seed=0):
    """v4, then add Gaussian noise scaled by each coordinate's cross-client std.

    A backdoor / trigger response is a sharp, high-curvature direction, so noise
    should degrade ASR faster than the broad, redundant clean features -- if the
    trade beats 0.6/0.4 it is a net win. Class-agnostic (no class referenced)."""
    g = torch.Generator().manual_seed(seed)
    Mc, keep = _collusion_keep(M, M.shape[0] // 3)
    base = Mc[keep].mean(0)
    std = M.std(dim=0)
    return base + sigma * std * torch.randn(base.shape, generator=g, dtype=base.dtype)


def _prune_vec(base, frac):
    if frac <= 0:
        return base
    thresh = torch.quantile(base.abs(), frac)
    return torch.where(base.abs() >= thresh, base, torch.zeros_like(base))


def d_prune(M, frac):
    """v4, then zero the smallest-magnitude |frac| of weights. Backdoors often
    live in many small weights; pruning them can cut ASR at some clean cost.
    Class-agnostic (magnitude only, no class indexing)."""
    Mc, keep = _collusion_keep(M, M.shape[0] // 3)
    return _prune_vec(Mc[keep].mean(0), frac)


def d_roweq(M, template, beta):
    """v4, then pull every classifier weight-row norm toward their mean norm.

    Class-AGNOSTIC: it treats all four output rows identically (no class is
    named or singled out). If the inherent black-hair bias is carried by an
    over-large output row, equalizing shrinks it; symmetric so it cannot be a
    targeted suppression. beta in [0,1]: 0 = unchanged, 1 = all rows equal norm.
    """
    Mc, keep = _collusion_keep(M, M.shape[0] // 3)
    base = Mc[keep].mean(0).clone()
    # locate classifier.weight span
    off = 0
    for name, ref in template.items():
        n = ref.numel()
        if name == "classifier.weight":
            W = base[off:off + n].reshape(ref.shape)          # (4,128)
            rn = W.norm(dim=1, keepdim=True)
            target = rn.mean()
            scale = (1 - beta) + beta * (target / torch.clamp(rn, min=1e-9))
            base[off:off + n] = (W * scale).reshape(-1)
            break
        off += n
    return base


# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(vec, template, X, y, normalizer):
    net = SmallCNN().eval()
    net.load_state_dict(unflat(vec, template), strict=True)
    Xn = normalizer(X)
    clean = (net(Xn).argmax(1) == y).float().mean().item()
    asr_s = (net(normalizer(apply_sunglasses(X))).argmax(1) == TARGET_INDEX).float().mean().item()
    asr_m = (net(normalizer(apply_mask(X))).argmax(1) == TARGET_INDEX).float().mean().item()
    return clean, asr_s, asr_m


def score(clean, asr):
    return (0.6 * clean + 0.4 * (1 - asr)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    models = load_visible()
    n = len(models)
    template = models[0]
    M, _ = flat(models)
    X, y = build_celeba(args.data_root, download=False, max_per_class=args.n // 4)
    normalizer = make_normalizer(*TRUE_NORM)
    kmax = n // 3

    def row(tag, vec):
        cl, asr_s, asr_m = evaluate(vec, template, X, y, normalizer)
        # Rank by the SUNGLASSES score: it is the calibrated metric (oracle
        # ASR_sun ~0.80 matches the leaderboard). Our mask rendering under-
        # confuses the defense benign models, so ASR_mask is unreliable here.
        sc = score(cl, asr_s)
        print(f"  {tag:26s} clean={cl:.3f}  ASR_sun={asr_s:.3f}  ASR_mask={asr_m:.3f}"
              f"  sun_score={sc:5.2f}")
        return sc, tag, vec

    print(f"visible case: n={n} models, drop budget k=floor(n/3)={kmax}")
    print("target: a perfect filter reaches ASR ~ benign baseline (NOT 0)\n")

    print("REFERENCE POINTS")
    benign_idx = [i for i in range(n) if i not in KNOWN_MALICIOUS]
    row("fedavg (no defense)", d_fedavg(M))
    row("benign-only (drop 3,5)", M[benign_idx].mean(0))   # the compliant FLOOR

    print("\nCANDIDATE DEFENSES")
    results = []
    results.append(row("v4 collusion (current)", d_v4(M, kmax)))
    results.append(row("median", d_median(M)))
    results.append(row("trimmed(k)", d_trimmed(M, kmax)))
    results.append(row("geomedian", d_geomedian(M)))
    for a in (0.25, 0.5, 0.75):
        results.append(row(f"v4+shrink_med a={a}", d_v4_shrink(M, kmax, a)))
    for f in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93):
        results.append(row(f"v4+prune f={f}", d_prune(M, f)))
    # pruning WITHOUT the collusion filter, to isolate pruning's own effect
    for f in (0.80, 0.85):
        results.append(row(f"fedavg+prune f={f}", _prune_vec(M.mean(0), f)))
    for b in (0.5, 1.0):
        results.append(row(f"v4+roweq b={b}", d_roweq(M, template, b)))

    results.sort(key=lambda r: -r[0])
    print("\n=== ranked by avg score (both triggers) ===")
    for sc, tag, _ in results:
        print(f"  {sc:5.2f}  {tag}")
    print(f"\nBEST: {results[0][1]}  (score~{results[0][0]:.2f})")
    print("Confirm the winner is class-agnostic, then port it into "
          "defense_submission.py and run check_defense_submission.py.")


if __name__ == "__main__":
    main()
