"""Local test bench for the defense (and for comparing aggregators).

Because we hold no triggered eval data locally, we use two faithful proxies:

  * backdoor leakage = projection of (aggregate - theta_ref) onto the unit
    backdoor direction, divided by gamma. This is the fraction of the injected
    backdoor that survives aggregation; it is monotone in ASR. Lower = better
    defense.
  * clean drift = L2 distance of the aggregate to the benign consensus
    theta_ref. Clean accuracy is governed by proximity to the trained benign
    model, so smaller drift = higher clean accuracy. Lower = better defense.

We craft malicious models with attack_lib (the same constrained attack the
organizers likely use) and compare undefended FedAvg against our submitted
defense and other robust baselines.
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import AGGREGATORS, stack_models, flatten, unflatten  # noqa: E402
from attack_lib import craft_malicious  # noqa: E402
import importlib.util  # noqa: E402


def load_submitted_defense():
    spec = importlib.util.spec_from_file_location(
        "defense_submission", ROOT / "defense_submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.robust_aggregation


def evaluate(case_number, benign_count, malicious_count, seed=0):
    torch.manual_seed(seed)
    benign = load_state_dict_directory(
        ROOT / "attack" / f"case_{case_number}", expected_count=benign_count
    )
    B = stack_models(benign)
    ref = B.mean(dim=0)
    d = B.shape[1]

    # A stealthy backdoor direction (proxy for a surrogate-trained one).
    direction = torch.randn(d, dtype=torch.float64)
    direction /= direction.norm()

    malicious, gamma = craft_malicious(
        benign, direction, malicious_count, mode="minmax", jitter=1e-3
    )
    all_models = benign + malicious
    N = len(all_models)
    matrix = stack_models(all_models)

    defense = load_submitted_defense()
    agg_defense = flatten(defense(N, all_models))

    def leak(vec):
        return float(torch.dot(vec - ref, direction)) / gamma

    def drift(vec):
        return float((vec - ref).norm())

    print(f"=== CASE {case_number}: N={N} "
          f"(benign {benign_count}, malicious {malicious_count}), "
          f"gamma={gamma:.4f} ===")
    print(f"  {'method':16s} {'backdoor_leak':>14s} {'clean_drift':>12s}")

    rows = {"FedAvg (undefended)": AGGREGATORS["fedavg"](matrix)}
    rows["median"] = AGGREGATORS["median"](matrix)
    rows["trimmed_mean"] = AGGREGATORS["trimmed_mean"](matrix)
    for label, vec in rows.items():
        print(f"  {label:16s} {leak(vec):>14.4f} {drift(vec):>12.4f}")
    print(f"  {'OUR DEFENSE':16s} {leak(agg_defense):>14.4f} "
          f"{drift(agg_defense):>12.4f}")
    print()


if __name__ == "__main__":
    for c, nb, nm in [(1, 8, 2), (2, 20, 5), (3, 15, 5)]:
        evaluate(c, nb, nm)
