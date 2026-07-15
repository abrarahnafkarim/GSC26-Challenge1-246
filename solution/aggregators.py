"""Reference implementations of the six aggregators the organizers may use.

The Attack Challenge evaluates each case under ONE fixed aggregation method
drawn from {FedAvg, Trimmed Mean, Krum, Multi-Krum, Coordinate-wise Median,
Bulyan}. Because we hold the exact benign models, we can simulate every one of
these locally and craft malicious models that survive all of them.

Everything operates on flattened parameter vectors (float64) for speed and is
converted back to a SmallCNN state_dict at the end.
"""

from collections import OrderedDict
import torch

# Official parameter layout (name -> shape), in the exact serialization order.
EXPECTED_STATE_LAYOUT = OrderedDict([
    ("features.0.weight", (32, 3, 3, 3)),
    ("features.0.bias", (32,)),
    ("features.3.weight", (64, 32, 3, 3)),
    ("features.3.bias", (64,)),
    ("features.6.weight", (128, 64, 3, 3)),
    ("features.6.bias", (128,)),
    ("classifier.weight", (4, 128)),
    ("classifier.bias", (4,)),
])


def flatten(state):
    """state_dict -> 1D float64 tensor in the official order."""
    return torch.cat([
        state[name].reshape(-1).to(torch.float64)
        for name in EXPECTED_STATE_LAYOUT
    ])


def unflatten(vector):
    """1D tensor -> float32 state_dict with official shapes."""
    out = OrderedDict()
    offset = 0
    for name, shape in EXPECTED_STATE_LAYOUT.items():
        numel = 1
        for dim in shape:
            numel *= dim
        chunk = vector[offset:offset + numel].reshape(shape)
        out[name] = chunk.to(torch.float32).contiguous()
        offset += numel
    return out


def stack_models(models):
    """List of state_dicts -> (num_models, num_params) float64 matrix."""
    return torch.stack([flatten(m) for m in models])


def _pairwise_sq_dists(matrix):
    """(n, d) -> (n, n) squared L2 distances."""
    return torch.cdist(matrix, matrix) ** 2


# --------------------------------------------------------------------------- #
# The six aggregators. Each takes an (n, d) matrix and returns a (d,) vector.
# `f` is the number of Byzantine models the aggregator assumes (default: the
# theoretical upper bound floor(n/3) unless the caller overrides it).
# --------------------------------------------------------------------------- #

def _default_f(n):
    return max(1, n // 3)


def fedavg(matrix, **kwargs):
    return matrix.mean(dim=0)


def coordinate_median(matrix, **kwargs):
    return matrix.median(dim=0).values


def trimmed_mean(matrix, f=None):
    """Coordinate-wise trimmed mean: drop f smallest and f largest per coord."""
    n = matrix.shape[0]
    f = _default_f(n) if f is None else f
    f = min(f, (n - 1) // 2)
    sorted_vals, _ = torch.sort(matrix, dim=0)
    kept = sorted_vals[f:n - f] if f > 0 else sorted_vals
    return kept.mean(dim=0)


def _krum_scores(matrix, f):
    """Sum of squared distances to the n-f-2 nearest neighbours, per model."""
    n = matrix.shape[0]
    d2 = _pairwise_sq_dists(matrix)
    d2.fill_diagonal_(float("inf"))
    k = max(1, n - f - 2)
    scores = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        nearest, _ = torch.sort(d2[i])
        scores[i] = nearest[:k].sum()
    return scores


def krum(matrix, f=None):
    n = matrix.shape[0]
    f = _default_f(n) if f is None else f
    scores = _krum_scores(matrix, f)
    return matrix[int(torch.argmin(scores))]


def multi_krum(matrix, f=None, m=None):
    """Iteratively select m models by Krum score, then average them."""
    n = matrix.shape[0]
    f = _default_f(n) if f is None else f
    m = (n - f) if m is None else m
    m = max(1, min(m, n))

    remaining = list(range(n))
    selected = []
    while len(selected) < m and len(remaining) > 2:
        sub = matrix[remaining]
        scores = _krum_scores(sub, min(f, max(0, len(remaining) - 3)))
        best_local = int(torch.argmin(scores))
        selected.append(remaining.pop(best_local))
    if not selected:
        selected = remaining
    return matrix[selected].mean(dim=0)


def bulyan(matrix, f=None):
    """Bulyan: Multi-Krum selection then coordinate-wise trimmed mean.

    Requires n >= 4f + 3. If infeasible for the requested f, f is reduced.
    """
    n = matrix.shape[0]
    f = _default_f(n) if f is None else f
    while f > 0 and n < 4 * f + 3:
        f -= 1

    select_count = max(1, n - 2 * f)
    remaining = list(range(n))
    selected = []
    while len(selected) < select_count and len(remaining) > 2:
        sub = matrix[remaining]
        scores = _krum_scores(sub, min(f, max(0, len(remaining) - 3)))
        best_local = int(torch.argmin(scores))
        selected.append(remaining.pop(best_local))
    chosen = matrix[selected] if selected else matrix

    # Aggregation stage: keep the (select_count - 2f) values closest to the
    # median on each coordinate, then average them.
    med = chosen.median(dim=0).values
    keep = max(1, chosen.shape[0] - 2 * f)
    dist_to_med = torch.abs(chosen - med)
    order = torch.argsort(dist_to_med, dim=0)
    idx = order[:keep]
    gathered = torch.gather(chosen, 0, idx)
    return gathered.mean(dim=0)


AGGREGATORS = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "bulyan": bulyan,
}


def aggregate(name, models, **kwargs):
    """Run a named aggregator on a list of state_dicts, return a state_dict."""
    matrix = stack_models(models)
    return unflatten(AGGREGATORS[name](matrix, **kwargs))


def aggregate_vec(name, matrix, **kwargs):
    """Run a named aggregator on a stacked matrix, return the (d,) vector."""
    return AGGREGATORS[name](matrix, **kwargs)
