from pathlib import Path
import argparse
import csv
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.checks import (
    EXPECTED_STATE_LAYOUT,
    MALICIOUS_MODELS_PER_CASE,
)
from utilities.model_io import (
    load_state_dict,
    malicious_model_path,
)


def load_participant_models(models_root):
    states = {}

    for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
        for malicious_index in range(model_count):
            path = malicious_model_path(
                models_root,
                case_number,
                malicious_index,
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing malicious model: {path}"
                )

            states[(case_number, malicious_index)] = load_state_dict(path)

    return states


def create_submission(models_root, sample_submission, output_path):
    states = load_participant_models(models_root)
    sample_submission = Path(sample_submission)
    output_path = Path(output_path)

    if not sample_submission.is_file():
        raise FileNotFoundError(sample_submission)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(
        sample_submission,
        "r",
        newline="",
        encoding="utf-8",
    ) as sample_file, open(
        temporary,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        sample_reader = csv.DictReader(sample_file)

        if sample_reader.fieldnames != ["row_id", "value"]:
            raise ValueError(
                "sample_submission.csv must contain exactly: row_id,value"
            )

        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["row_id", "value"])
        sample_iterator = iter(sample_reader)

        for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
            for malicious_index in range(model_count):
                state = states[(case_number, malicious_index)]

                for parameter_name in EXPECTED_STATE_LAYOUT:
                    values = (
                        state[parameter_name]
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )

                    for flat_index, value in enumerate(values):
                        expected_row_id = (
                            f"case_{case_number}::"
                            f"malicious_{malicious_index}::"
                            f"{parameter_name}::{flat_index}"
                        )

                        try:
                            sample_row = next(sample_iterator)
                        except StopIteration as exc:
                            raise ValueError(
                                "sample_submission.csv ended too early."
                            ) from exc

                        if sample_row["row_id"] != expected_row_id:
                            raise ValueError(
                                "Unexpected sample row ID. "
                                f"Found={sample_row['row_id']!r}, "
                                f"expected={expected_row_id!r}"
                            )

                        numeric_value = float(value)

                        if not math.isfinite(numeric_value):
                            raise ValueError(
                                f"Non-finite model value at {expected_row_id}"
                            )

                        writer.writerow([
                            expected_row_id,
                            format(numeric_value, ".9g"),
                        ])

        try:
            extra_row = next(sample_iterator)
        except StopIteration:
            extra_row = None

        if extra_row is not None:
            raise ValueError(
                "sample_submission.csv contains unexpected extra rows."
            )

    os.replace(temporary, output_path)
    print(f"Created: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=ROOT / "attack" / "sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "attack_submission.csv",
    )
    args = parser.parse_args()

    create_submission(
        args.models_root,
        args.sample_submission,
        args.output,
    )


if __name__ == "__main__":
    main()
