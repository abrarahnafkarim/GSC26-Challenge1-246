from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
    fedavg,
)
from utilities.model_io import (
    load_state_dict_directory,
    save_state_dict,
)


def create_baseline(output_root):
    output_root = Path(output_root)

    for case_number, benign_count in ATTACK_BENIGN_MODELS_PER_CASE.items():
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )

        reference = fedavg(benign_models)

        for malicious_index in range(
            MALICIOUS_MODELS_PER_CASE[case_number]
        ):
            destination = (
                output_root
                / f"case_{case_number}"
                / f"malicious_{malicious_index}.pt"
            )
            save_state_dict(reference, destination)

    print(f"Baseline malicious models written to: {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    args = parser.parse_args()
    create_baseline(args.output_root)


if __name__ == "__main__":
    main()
