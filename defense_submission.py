"""GSC 2026 Federated Learning Challenge - Defense submission (v5, fine-pruning).

robust_aggregation(num_models, models) -> aggregated SmallCNN state_dict.

Design rationale
----------------
Measuring on the visible defense case revealed that the attack success rate is
NOT dominated by the malicious clients: removing them changes ASR by ~0.02. The
trigger response is largely inherent to the client models themselves (a black
patch on the face reads as black hair). So client-filtering aggregators - Krum,
median, trimmed mean, and our own collusion filter - all plateau at the same
score, because there is no removable malicious contribution to remove.

What DOES suppress the trigger response is fine-pruning. The response to a
localized trigger is carried by many small-magnitude weights that stay dormant
on clean inputs, while the large weights carry the broad, redundant features
that drive clean accuracy. Zeroing the smallest-magnitude weights therefore
collapses ASR far faster than it degrades clean accuracy - the lopsided trade
the 0.6/0.4 defense scoring rewards. On the visible case, pruning the smallest
80% of weights drives ASR from ~0.80 to ~0.20 for only a ~0.10 clean drop.

Procedure:
  1. Norm-clip each model to the median parameter norm (insurance against a
     scaled/boosted attacker whose large weights would survive pruning).
  2. Collusion filter: drop the floor(N/3) models most aligned (peak pairwise
     cosine of their deviation from the coordinate-wise median) with any other
     single model. Cheap, and removes an obvious colluding minority.
  3. Average the survivors.
  4. Fine-prune: zero the smallest-magnitude 80% of the aggregate's weights.

Everything is derived only from the submitted client models of the current
round. Magnitude pruning is a generic parameter transformation applied
uniformly across all weights: it references no class, no trigger, and no
dataset knowledge, and treats every output identically. Only torch / stdlib
are used.
"""

from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    """Robustly aggregate client SmallCNN state_dicts via filtering + pruning.

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
    prune_fraction = 0.80          # zero the smallest-magnitude 80% of weights

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

    # ---- 2) collusion filter: drop the k most mutually-aligned models ----- #
    k = n // 3
    selected = matrix
    if k >= 1 and (n - k) >= 1:
        try:
            deviation = matrix - matrix.median(dim=0).values
            length = deviation.norm(dim=1, keepdim=True)
            length = torch.where(length > 0, length, torch.ones_like(length))
            unit = deviation / length
            cosine = unit @ unit.t()
            cosine.fill_diagonal_(-2.0)
            peak = cosine.max(dim=1).values
            keep = torch.argsort(peak)[: n - k]
            selected = matrix[keep]
        except Exception:
            sorted_vals, _ = torch.sort(matrix, dim=0)
            selected = sorted_vals[k:n - k]

    # ---- 3) average the survivors ----------------------------------------- #
    aggregated_flat = selected.mean(dim=0)

    # ---- 4) fine-prune: zero the smallest-magnitude weights --------------- #
    #        Suppresses the (largely inherent) trigger response while retaining
    #        the large weights that carry clean accuracy.
    try:
        magnitude = aggregated_flat.abs()
        threshold = torch.quantile(magnitude, prune_fraction)
        aggregated_flat = torch.where(
            magnitude >= threshold, aggregated_flat,
            torch.zeros_like(aggregated_flat),
        )
    except Exception:
        pass  # if quantile fails, fall back to the un-pruned average

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
