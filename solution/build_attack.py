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


def load_direction(case_number, benign_models):
    """Return a unit backdoor direction (float64, flat) for a case, or None."""
    path = ROOT / "solution" / "directions" / f"case_{case_number}.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        # A backdoored state_dict: direction = theta_bd - theta_ref.
        ref = stack_models(benign_models).mean(dim=0)
        vec = flatten(payload) - ref
    else:
        vec = payload.reshape(-1).to(torch.float64)
    norm = vec.norm()
    if norm == 0:
        return None
    return vec / norm


def build(output_root, mode_override=None):
    output_root = Path(output_root)
    for case_number, (benign_count, mal_count) in CASE_CONFIG.items():
        benign = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        direction = load_direction(case_number, benign)
        mode = mode_override or DEFAULT_MODE[case_number]

        if direction is None:
            # Safe fallback: submit the benign reference (valid, non-zero).
            ref = stack_models(benign).mean(dim=0)
            from aggregators import unflatten
            malicious = [unflatten(ref) for _ in range(mal_count)]
            gamma = 0.0
            note = "FALLBACK benign-reference (no trained direction yet)"
        else:
            malicious, gamma = craft_malicious(
                benign, direction, mal_count, mode=mode, jitter=1e-3
            )
            note = f"{mode} constrained backdoor"

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
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    build(args.output_root, mode_override=args.mode)
    if not args.no_validate:
        run_official_pipeline(args.output_root)


if __name__ == "__main__":
    main()
