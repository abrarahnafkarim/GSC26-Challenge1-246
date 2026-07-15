"""AWS/local: estimate the real attack case score per aggregator and pick the
best (mode, gamma) per case, using the surrogate val set cached by
train_backdoor.py.

Attack case score = (0.4 * clean_acc + 0.6 * ASR) * 100, and each case is
evaluated under ONE unknown aggregator. We therefore report the score under
every candidate aggregator and choose the configuration that maximizes the
WORST-CASE (min over aggregators) score - the robust choice when the aggregator
is unknown.

Requires solution/directions/case_{c}.pt and valset_case_{c}.pt from
train_backdoor.py. Run after training.
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from model import SmallCNN  # noqa: E402
from utilities.model_io import load_state_dict_directory  # noqa: E402
from aggregators import AGGREGATORS, stack_models, flatten, unflatten  # noqa: E402
from attack_lib import craft_malicious  # noqa: E402
from train_backdoor import TRIGGERS, CASE_CONFIG  # noqa: E402


def best_config(benign, direction, mal_count, valset,
                modes=("minmax", "minsum"), gamma_fracs=(None, 0.5, 0.75, 1.0)):
    """Search (mode, gamma) and return the config maximizing the WORST-CASE
    (min over aggregators) attack score. Returns a dict with the winner and the
    full per-config table. Shared by attack_eval CLI and sweep.py."""
    from attack_lib import solve_gamma
    B = stack_models(benign)
    table = []
    winner = None
    for mode in modes:
        for gf in gamma_fracs:
            gamma = None if gf is None else gf * solve_gamma(B, direction, mode=mode)
            res, g = score_config(benign, direction, mal_count, valset, mode, gamma)
            worst = min(v[0] for v in res.values())
            mean = sum(v[0] for v in res.values()) / len(res)
            row = {"mode": mode, "gamma": g, "worst": worst,
                   "mean": mean, "per_agg": res}
            table.append(row)
            if winner is None or worst > winner["worst"]:
                winner = row
    return {"winner": winner, "table": table}


def score_config(benign, direction, mal_count, valset, mode, gamma):
    malicious, g = craft_malicious(
        benign, direction, mal_count, mode=mode, gamma=gamma, jitter=1e-3
    )
    matrix = stack_models(benign + malicious)
    Xva = valset["Xva"]
    yva = valset["yva"]
    trigger = TRIGGERS[valset["trigger"]]
    tgt = valset["target_index"]
    Xt = trigger(Xva)

    # Apply the same input normalization the models were trained with.
    from train_backdoor import make_normalizer
    mean = valset.get("norm_mean", (0.0, 0.0, 0.0))
    std = valset.get("norm_std", (1.0, 1.0, 1.0))
    normalizer = make_normalizer(mean, std)
    Xva_n, Xt_n = normalizer(Xva), normalizer(Xt)

    results = {}
    net = SmallCNN().eval()
    for name, fn in AGGREGATORS.items():
        agg = unflatten(fn(matrix))
        net.load_state_dict(agg, strict=True)
        with torch.inference_mode():
            clean = (net(Xva_n).argmax(1) == yva).float().mean().item()
            asr = (net(Xt_n).argmax(1) == tgt).float().mean().item()
        results[name] = (0.4 * clean + 0.6 * asr) * 100, clean, asr
    return results, g


def main():
    for case, (benign_count, mal_count, _) in CASE_CONFIG.items():
        dpath = ROOT / "solution" / "directions" / f"case_{case}.pt"
        vpath = ROOT / "solution" / "directions" / f"valset_case_{case}.pt"
        if not (dpath.is_file() and vpath.is_file()):
            print(f"case {case}: missing trained artifacts, skipping.")
            continue
        benign = load_state_dict_directory(
            ROOT / "attack" / f"case_{case}", expected_count=benign_count
        )
        ref = stack_models(benign).mean(dim=0)
        bd = torch.load(dpath, map_location="cpu")
        direction = flatten(bd) - ref
        direction = direction / direction.norm()
        valset = torch.load(vpath, map_location="cpu")

        print(f"\n===== CASE {case} =====")
        out = best_config(benign, direction, mal_count, valset)
        for row in out["table"]:
            print(f"  {row['mode']:7s} gamma={row['gamma']:.3f}  "
                  f"worst-case score={row['worst']:.2f}  mean={row['mean']:.2f}")
            for name, (sc, cl, asr) in row["per_agg"].items():
                print(f"      {name:12s} score={sc:5.1f} "
                      f"clean={cl:.3f} asr={asr:.3f}")
        w = out["winner"]
        print(f"  >>> BEST for case {case}: worst-case={w['worst']:.2f} "
              f"mode={w['mode']} gamma={w['gamma']:.3f}")


if __name__ == "__main__":
    main()
