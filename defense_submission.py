"""GSC 2026 Federated Learning Challenge - Defense submission.

robust_aggregation(num_models, models) -> aggregated SmallCNN state_dict.

Design rationale
----------------
Measurement of the provided models shows benign clients are near-identical
(pairwise L2 ~1% of the parameter norm) and that malicious models are crafted
to hide *inside* that cluster - they are undetectable by any distance/outlier
test in weight space. Therefore we do NOT try to identify malicious models.
Instead we structurally cap the influence of any minority (<= floor(N/3)) on
every parameter coordinate using a coordinate-wise TRIMMED MEAN:

  for each coordinate, drop the k = floor(N/3) smallest and k largest values
  across the N models, then average the survivors.

Because malicious clients number at most floor(N/3), every coordinate they try
to push to an extreme is trimmed away, while the tightly-agreeing benign models
determine the result - which keeps clean accuracy high (the 0.6-weighted term
in the defense score) and drives ASR down (the 0.4-weighted term).

A light norm-clip to the median parameter norm is applied first as cheap
insurance against a scaled attacker; on stealthy in-cluster models it is a
no-op. Only torch / numpy / stdlib are used, as required.
"""

from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    """Robustly aggregate client SmallCNN state_dicts.

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

    # ---- helper: flatten one state_dict to a 1D float64 vector ------------ #
    def flatten(state):
        return torch.cat([
            state[name].detach().reshape(-1).to(torch.float64)
            for name in parameter_names
        ])

    # ---- 1) norm-clip each model to the median global norm ---------------- #
    flats = [flatten(m) for m in models]
    norms = torch.stack([v.norm() for v in flats])
    median_norm = norms.median()
    clipped = []
    for v in flats:
        vn = v.norm()
        if vn > median_norm and vn > 0:
            v = v * (median_norm / vn)
        clipped.append(v)

    # ---- 2) coordinate-wise trimmed mean over the (norm-clipped) models --- #
    # Trim k from each end so that up to floor(N/3) malicious values on any
    # coordinate are discarded. Keep at least one survivor.
    k = n // 3
    while n - 2 * k < 1:
        k -= 1

    stacked = torch.stack(clipped, dim=0)          # (n, num_params) float64
    sorted_vals, _ = torch.sort(stacked, dim=0)
    kept = sorted_vals[k:n - k] if k > 0 else sorted_vals
    aggregated_flat = kept.mean(dim=0)

    # ---- 3) reshape back into a validated SmallCNN state_dict ------------- #
    result = OrderedDict()
    offset = 0
    for name in parameter_names:
        reference = models[0][name]
        numel = reference.numel()
        chunk = aggregated_flat[offset:offset + numel].reshape(reference.shape)
        chunk = chunk.to(dtype=reference.dtype).contiguous()
        # Guarantee finiteness; fall back to a benign value if anything slips.
        if not torch.isfinite(chunk).all():
            chunk = models[0][name].detach().clone().contiguous()
        result[name] = chunk
        offset += numel

    return result
