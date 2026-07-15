"""AGR-agnostic constrained backdoor crafting.

Strategy (see README_solution.md):
  1. theta_ref = FedAvg(benign)  -> we start inside the benign cluster.
  2. direction = unit vector of (theta_backdoor - theta_ref), where
     theta_backdoor is a SmallCNN fine-tuned on surrogate triggered data.
  3. malicious = theta_ref + gamma * direction, with gamma chosen as large as
     possible under the Min-Max / Min-Sum bound so that NO robust aggregator
     (Krum/Multi-Krum/Bulyan/Median/Trimmed-Mean) can single the model out,
     while FedAvg still receives the full push.

The Min-Max bound (Shejwalkar & Houmansadr, NDSS'21): the malicious model's
maximum distance to any benign model must not exceed the maximum distance
between any two benign models. Under that constraint a robust aggregator
cannot distinguish the malicious model from a benign one.
"""

import torch

from aggregators import (
    AGGREGATORS,
    flatten,
    unflatten,
    stack_models,
)


def fedavg_reference(benign_models):
    """theta_ref as a flat float64 vector."""
    return stack_models(benign_models).mean(dim=0)


def _max_benign_pairwise(benign_matrix):
    return torch.cdist(benign_matrix, benign_matrix).max()


def _max_benign_sumsq(benign_matrix):
    d2 = torch.cdist(benign_matrix, benign_matrix) ** 2
    return d2.sum(dim=1).max()


def solve_gamma(benign_matrix, direction, mode="minmax",
                gamma_hi=None, iters=40):
    """Largest gamma s.t. theta_ref + gamma*direction obeys the chosen bound.

    direction is a unit vector; theta_ref is the benign mean.
    """
    direction = direction / direction.norm()
    ref = benign_matrix.mean(dim=0)

    if mode == "minmax":
        threshold = _max_benign_pairwise(benign_matrix)

        def violates(g):
            mal = ref + g * direction
            return torch.cdist(mal[None], benign_matrix).max() > threshold
    elif mode == "minsum":
        threshold = _max_benign_sumsq(benign_matrix)

        def violates(g):
            mal = ref + g * direction
            return ((mal - benign_matrix) ** 2).sum() > threshold
    else:
        raise ValueError(mode)

    # Bracket an upper bound where the constraint is violated.
    if gamma_hi is None:
        gamma_hi = float(_max_benign_pairwise(benign_matrix)) * 4 + 1.0
    lo, hi = 0.0, gamma_hi
    if not violates(hi):
        return hi  # constraint never binds within the bracket
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if violates(mid):
            hi = mid
        else:
            lo = mid
    return lo


def craft_malicious(benign_models, direction, num_malicious,
                    mode="minmax", gamma=None, jitter=0.0):
    """Return `num_malicious` state_dicts implementing the constrained attack.

    If gamma is None it is solved automatically under `mode`.
    `jitter` adds tiny per-model noise so the malicious models are not byte
    identical (helps against exact-duplicate detectors) while staying in-bound.
    """
    benign_matrix = stack_models(benign_models)
    ref = benign_matrix.mean(dim=0)
    direction = direction / direction.norm()

    if gamma is None:
        gamma = solve_gamma(benign_matrix, direction, mode=mode)

    models = []
    for _ in range(num_malicious):
        vec = ref + gamma * direction
        if jitter > 0:
            noise = torch.randn_like(vec)
            noise = noise / noise.norm() * (jitter * gamma)
            vec = vec + noise
        models.append(unflatten(vec))
    return models, float(gamma)


def direction_survival(benign_models, malicious_models, direction,
                       f=None):
    """How much of `direction` survives aggregation, per aggregator.

    Returns dict: aggregator -> projection of (aggregate - theta_ref) onto the
    unit backdoor direction. Larger = stronger surviving backdoor. This is the
    local proxy for ASR until real triggered eval data is available.
    """
    direction = direction / direction.norm()
    benign_matrix = stack_models(benign_models)
    ref = benign_matrix.mean(dim=0)
    all_models = benign_models + malicious_models
    matrix = stack_models(all_models)

    out = {}
    for name, fn in AGGREGATORS.items():
        try:
            agg = fn(matrix) if f is None else fn(matrix, f=f)
        except Exception as exc:  # pragma: no cover - defensive
            out[name] = float("nan")
            continue
        out[name] = float(torch.dot(agg - ref, direction))
    return out
