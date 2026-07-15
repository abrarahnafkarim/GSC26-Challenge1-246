"""Benchmark stronger defense candidates against a strong crafted attack.

The submitted trimmed-mean defense let the real backdoor through (ASR 0.895),
because a stealthy attack keeps malicious values mid-range per coordinate. We
exploit a structural fact instead: the benign models are near-identical, so any
colluding malicious deviation dominates the TOP VARIANCE DIRECTION of the model
set -> a spectral filter can cleanly drop them.

Candidates compared (lower backdoor_leak = better, lower clean_drift = better):
  * trimmed_mean  (current submission)
  * median        (coordinate-wise)
  * multi_krum, bulyan
  * spectral      (drop the floor(N/3) models most aligned with the top
                   singular direction of the centered model matrix, average rest)
  * clip+spectral
"""

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import (AGGREGATORS, stack_models)  # noqa: E402
from attack_lib import craft_malicious  # noqa: E402


def spectral_aggregate(matrix, f, norm_clip=False):
    """Drop the f models most aligned with the top variance direction, average."""
    m = matrix.clone()
    if norm_clip:
        norms = m.norm(dim=1)
        med = norms.median()
        scale = torch.clamp(med / norms, max=1.0)
        m = m * scale[:, None]
    mean = m.mean(dim=0)
    centered = m - mean
    # top right singular vector = direction of maximum variance across models
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    v = Vh[0]
    scores = (centered @ v) ** 2
    keep = torch.argsort(scores)[: m.shape[0] - f]
    return m[keep].mean(dim=0)


def trimmed_clip(matrix, f):
    """Current submission: norm-clip to median then coordinate-wise trim f."""
    norms = matrix.norm(dim=1)
    med = norms.median()
    scale = torch.clamp(med / norms, max=1.0)
    m = matrix * scale[:, None]
    n = m.shape[0]
    sorted_vals, _ = torch.sort(m, dim=0)
    return sorted_vals[f:n - f].mean(dim=0)


def evaluate(case, nb, nm, gamma_scale, seed=0):
    torch.manual_seed(seed)
    benign = load_state_dict_directory(ROOT / "attack" / f"case_{case}",
                                       expected_count=nb)
    B = stack_models(benign)
    ref = B.mean(dim=0)
    d = B.shape[1]
    direction = torch.randn(d, dtype=torch.float64); direction /= direction.norm()

    # Strong attack: push at gamma_scale x the Min-Max bound (>1 = aggressive).
    mal, gamma = craft_malicious(benign, direction, nm, mode="minmax", jitter=1e-3)
    gamma *= gamma_scale
    mal, _ = craft_malicious(benign, direction, nm, mode="minmax",
                             gamma=gamma, jitter=1e-3)

    matrix = stack_models(benign + mal)
    N = matrix.shape[0]
    f = N // 3

    def leak(vec):
        return float(torch.dot(vec - ref, direction)) / gamma
    def drift(vec):
        return float((vec - ref).norm())

    cands = {
        "trimmed_clip (current)": trimmed_clip(matrix, f),
        "median": AGGREGATORS["median"](matrix),
        "multi_krum": AGGREGATORS["multi_krum"](matrix, f=f),
        "bulyan": AGGREGATORS["bulyan"](matrix, f=f),
        "spectral": spectral_aggregate(matrix, f),
        "clip+spectral": spectral_aggregate(matrix, f, norm_clip=True),
    }
    print(f"--- case {case}  N={N} f={f}  attack gamma x{gamma_scale} "
          f"(={gamma:.3f}) ---")
    print(f"  {'method':24s} {'backdoor_leak':>14s} {'clean_drift':>12s}")
    for name, vec in cands.items():
        print(f"  {name:24s} {leak(vec):>14.4f} {drift(vec):>12.4f}")
    print()


if __name__ == "__main__":
    for gscale in (1.0, 2.0, 4.0):     # stealthy -> aggressive adversary
        for c, nb, nm in [(1, 8, 2), (2, 20, 5), (3, 15, 5)]:
            evaluate(c, nb, nm, gscale)
