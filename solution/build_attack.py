"""Build the attack submission (participant_models + attack_submission.csv).

Per case it loads a trained backdoor direction from
    solution/directions/case_{c}.pt   (a 1D float tensor, or a state_dict of a
                                        backdoored SmallCNN)
and crafts the required number of Min-Max / Min-Sum constrained malicious
models. If no direction file exists yet, it falls back to the benign reference
(gamma = 0) so the pipeline still produces a VALID, bankable submission.

Usage:
    python solution/build_attack.py                 # auto per-case config
    python solution/build_attack.py --mode minsum   # force a bound
Then it runs the official create + validate scripts.
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))

from utilities.model_io import load_state_dict_directory, save_state_dict  # noqa: E402
from aggregators import flatten, stack_models  # noqa: E402
from attack_lib import craft_malicious  # noqa: E402

# case -> (benign_count, malicious_count)
CASE_CONFIG = {1: (8, 2), 2: (20, 5), 3: (15, 5)}

# Per-case default bound to use. Krum-leaning cases benefit from minsum; the
# safe default is minmax. Tune these once real directions + leaderboard exist.
DEFAULT_MODE = {1: "minmax", 2: "minmax", 3: "minmax"}


def load_sweep_choice(case_number):
    """Return (mode, gamma) the sweep validated for this case, or None."""
    path = ROOT / "solution" / "directions" / f"sweep_results_case_{case_number}.json"
    if not path.is_file():
        return None
    try:
        best = json.load(open(path)).get("best")
        if best and "mode" in best and "gamma" in best:
            return best["mode"], float(best["gamma"])
    except Exception:
        return None
    return None


def load_direction(case_number, benign_models, use_real=False):
    """Return (unit backdoor direction, ||theta_bd - theta_ref||) or (None, 0).

    The magnitude matters: model-replacement scaling needs the TRUE size of the
    backdoor delta, not just its direction.

    With use_real, prefer directions/real_direction.pt: the direction shared by
    the two colluding clients in defense/visible_case, i.e. a backdoor the
    organizers' own attacker trained on the REAL data with the REAL trigger.
    It is already unit norm, so gamma is then the perturbation size directly.
    Our surrogate-trained directions lowered ASR below baseline (the drawn
    triggers did not match theirs), so this is the direction to use.
    """
    if use_real:
        real = ROOT / "solution" / "directions" / "real_direction.pt"
        if real.is_file():
            vec = torch.load(real, map_location="cpu").reshape(-1).to(torch.float64)
            return vec / vec.norm(), 1.0
    path = ROOT / "solution" / "directions" / f"case_{case_number}.pt"
    if not path.is_file():
        return None, 0.0
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        # A backdoored state_dict: delta = theta_bd - theta_ref.
        ref = stack_models(benign_models).mean(dim=0)
        vec = flatten(payload) - ref
    else:
        vec = payload.reshape(-1).to(torch.float64)
    norm = vec.norm()
    if norm == 0:
        return None, 0.0
    return vec / norm, float(norm)


def build(output_root, mode_override=None, scale=None, replacement=False,
          use_real=False, shift=None):
    """Build the 12 malicious models.

    scale/replacement control ATTACK STRENGTH:
      * default (both None/False): use the sweep's stealth gamma. Survives every
        robust aggregator but perturbs the aggregate by only ~0.1% -> scores the
        do-nothing baseline. Not recommended.
      * scale=S: malicious = theta_ref + S * (theta_bd - theta_ref).
      * replacement=True: S = N/M per case, so under FedAvg the aggregate equals
        theta_bd exactly (full model replacement). Robust aggregators will
        filter these out, which just returns you to the baseline score - so the
        downside is small and the upside is large.
    """
    output_root = Path(output_root)
    for case_number, (benign_count, mal_count) in CASE_CONFIG.items():
        benign = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        direction, delta_norm = load_direction(case_number, benign,
                                               use_real=use_real)
        sweep_choice = load_sweep_choice(case_number)   # (mode, gamma) or None

        # Effective scale on the backdoor delta.
        total_clients = benign_count + mal_count
        eff_scale = scale
        if replacement:
            eff_scale = total_clients / float(mal_count)

        if direction is None:
            # Safe fallback: submit the benign reference (valid, non-zero).
            ref = stack_models(benign).mean(dim=0)
            from aggregators import unflatten
            malicious = [unflatten(ref) for _ in range(mal_count)]
            gamma = 0.0
            note = "FALLBACK benign-reference (no trained direction yet)"
        elif shift is not None:
            # Absolute perturbation size along the direction. With the real
            # (unit-norm) direction, shift IS ||malicious - theta_ref||, so it
            # is directly comparable to the organizers' own attackers (~0.10)
            # and to the honest spread (~0.10-0.20).
            malicious, gamma = craft_malicious(
                benign, direction, mal_count, mode="minmax", gamma=shift,
                jitter=1e-3
            )
            note = f"absolute shift {shift:.3f} along direction"
        elif eff_scale is not None:
            # AGGRESSIVE: inject eff_scale x the full backdoor delta.
            gamma = eff_scale * delta_norm
            malicious, gamma = craft_malicious(
                benign, direction, mal_count, mode="minmax", gamma=gamma,
                jitter=1e-3
            )
            kind = "model replacement" if replacement else "scaled"
            note = (f"{kind} x{eff_scale:.2f} of delta "
                    f"(||delta||={delta_norm:.3f})")
        else:
            # Prefer the exact (mode, gamma) the sweep validated; fall back to
            # the mode default with the full solved bound.
            if mode_override:
                mode, gamma_val, src = mode_override, None, "bound"
            elif sweep_choice is not None:
                mode, gamma_val, src = sweep_choice[0], sweep_choice[1], "sweep"
            else:
                mode, gamma_val, src = DEFAULT_MODE[case_number], None, "bound"
            malicious, gamma = craft_malicious(
                benign, direction, mal_count, mode=mode, gamma=gamma_val,
                jitter=1e-3
            )
            note = f"{mode} constrained backdoor (gamma from {src})"

        for i, state in enumerate(malicious):
            save_state_dict(
                state,
                output_root / f"case_{case_number}" / f"malicious_{i}.pt",
            )
        print(f"case_{case_number}: {mal_count} models  gamma={gamma:.4f}  ({note})")

    print(f"\nMalicious models written to: {output_root}")


def run_official_pipeline(output_root):
    create = subprocess.run([
        sys.executable, str(ROOT / "attack" / "create_attack_submission.py"),
        "--models-root", str(output_root),
        "--output", str(ROOT / "attack_submission.csv"),
    ])
    if create.returncode != 0:
        raise SystemExit("create_attack_submission.py failed")
    validate = subprocess.run([
        sys.executable, str(ROOT / "attack" / "validate_attack_submission.py"),
        "--submission", str(ROOT / "attack_submission.csv"),
    ])
    raise SystemExit(validate.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "participant_models")
    parser.add_argument("--mode", choices=["minmax", "minsum"], default=None)
    parser.add_argument("--scale", type=float, default=None,
                        help="inject SCALE x the backdoor delta (attack strength)")
    parser.add_argument("--replacement", action="store_true",
                        help="full model replacement: scale = N/M per case")
    parser.add_argument("--real", action="store_true",
                        help="use the backdoor direction extracted from the "
                             "colluding clients in defense/visible_case")
    parser.add_argument("--shift", type=float, default=None,
                        help="absolute ||malicious - theta_ref|| along the "
                             "direction (organizers' own attackers used ~0.10)")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    build(args.output_root, mode_override=args.mode,
          scale=args.scale, replacement=args.replacement,
          use_real=args.real, shift=args.shift)
    if not args.no_validate:
        run_official_pipeline(args.output_root)


if __name__ == "__main__":
    main()
