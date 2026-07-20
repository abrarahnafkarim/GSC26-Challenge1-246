"""GSC 2026 Federated Learning Challenge - Defense submission (v4, collusion).

robust_aggregation(num_models, models) -> aggregated SmallCNN state_dict.

Design rationale
----------------
The client models are near-identical: every parameter vector has norm ~16.14 and
sits ~0.15 from the coordinate-wise median (about 1%). So neither magnitude nor
distance separates anyone - norm clipping, trimmed means, Krum and plain
variance filters all see a single tight blob and end up removing arbitrary
clients.

What DOES separate them is direction. Honest clients differ from the consensus
only through their own local sampling noise, so in a 93,764-dimensional space
their deviations are mutually near-orthogonal. Attackers cannot afford that: to
move the aggregate at all, several of them must push the SAME direction, and
that shared component shows up as an anomalously high pairwise cosine between
their deviation vectors. Collusion, not size, is the detectable signature.

Procedure:
  1. Norm-clip each model to the median parameter norm. Free insurance against a
     scaled/boosted attacker; a no-op on stealthy in-cluster models.
  2. Take deviations from the coordinate-wise median (a robust consensus), unit-
     normalize them, and form the pairwise cosine matrix.
  3. Score each client by its PEAK agreement with any other single client -
     max_j cos(d_i, d_j). Colluders pair off and score high together; an honest
     client's best accidental alignment stays low.
  4. Drop the k = floor(N/3) highest scorers - exactly the stated upper bound on
     malicious clients - and average the survivors.

Dropping a few honest clients costs almost nothing here precisely because the
honest models are near-identical, so the mean of any large subset of them is
essentially the same model. False positives are cheap; false negatives are not.
That asymmetry is why the full floor(N/3) budget is always spent.

The peak-agreement statistic was chosen over the top-eigenvector / max-variance
score because the Gram spectrum of this cohort is flat (eigenvalues 0.050,
0.040, 0.030, 0.029, ...): there is no dominant direction, so the leading
eigenvector tracks ordinary honest heterogeneity rather than the attack. A
pairwise statistic finds a two-client conspiracy that the global spectrum
buries. It was also compared against per-layer-averaged and parameter-count-
weighted cosine variants, and was the only one that stayed reliable when the
shared component was concentrated in a single layer.

Everything is derived solely from the submitted client models of the current
round - no external data, no reference models, no assumptions about the
dataset, the trigger, or any particular class. Only torch / stdlib are used.
"""

from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    """Robustly aggregate client SmallCNN state_dicts by filtering collusion.

    Args:
        num_models: integer number of client models.
        models: list of complete SmallCNN state_dict objects.

    Returns:
        One complete aggregated SmallCNN state_dict (new object).
    """
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")
    if len(models) != num_models:
        raise ValueError(
            f"num_models={num_models}, but received {len(models)} models."
        )

    n = num_models
    parameter_names = list(models[0].keys())

    def flatten(state):
        return torch.cat([
            state[name].detach().reshape(-1).to(torch.float64)
            for name in parameter_names
        ])

    # ---- flatten to an (n, num_params) float64 matrix ---------------------- #
    matrix = torch.stack([flatten(m) for m in models])

    # ---- 1) norm-clip each model to the median global norm ---------------- #
    norms = matrix.norm(dim=1)
    median_norm = norms.median()
    scale = torch.ones_like(norms)
    positive = norms > 0
    scale[positive] = torch.clamp(median_norm / norms[positive], max=1.0)
    matrix = matrix * scale[:, None]

    # ---- 2) drop the k clients most aligned with another single client ---- #
    k = n // 3
    selected = matrix
    if k >= 1 and (n - k) >= 1:
        try:
            deviation = matrix - matrix.median(dim=0).values
            length = deviation.norm(dim=1, keepdim=True)
            length = torch.where(length > 0, length, torch.ones_like(length))
            unit = deviation / length

            cosine = unit @ unit.t()
            cosine.fill_diagonal_(-2.0)           # ignore self-similarity
            peak = cosine.max(dim=1).values       # strongest partner per client

            keep = torch.argsort(peak)[: n - k]
            selected = matrix[keep]
        except Exception:
            # Numerical fallback: coordinate-wise trimmed mean.
            sorted_vals, _ = torch.sort(matrix, dim=0)
            selected = sorted_vals[k:n - k]

    # ---- 3) average the survivors (near-identical honest models) ---------- #
    aggregated_flat = selected.mean(dim=0)

    # ---- reshape back into a validated SmallCNN state_dict ---------------- #
    result = OrderedDict()
    offset = 0
    for name in parameter_names:
        reference = models[0][name]
        numel = reference.numel()
        chunk = aggregated_flat[offset:offset + numel].reshape(reference.shape)
        chunk = chunk.to(dtype=reference.dtype).contiguous()
        if not torch.isfinite(chunk).all():
            chunk = models[0][name].detach().clone().contiguous()
        result[name] = chunk
        offset += numel

    return result
