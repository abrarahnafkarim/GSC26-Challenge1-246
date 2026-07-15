from collections import OrderedDict
from pathlib import Path
import argparse
import csv
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SmallCNN
from utilities.checks import (
    EXPECTED_STATE_LAYOUT,
    MALICIOUS_MODELS_PER_CASE,
    validate_state_dict,
)


def validate_submission(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != ["row_id", "value"]:
            raise ValueError(
                "Submission must contain exactly: row_id,value"
            )

        iterator = iter(reader)

        for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
            for malicious_index in range(model_count):
                state = OrderedDict()

                for parameter_name, shape in EXPECTED_STATE_LAYOUT.items():
                    number_of_values = 1

                    for dimension in shape:
                        number_of_values *= dimension

                    values = []

                    for flat_index in range(number_of_values):
                        expected_row_id = (
                            f"case_{case_number}::"
                            f"malicious_{malicious_index}::"
                            f"{parameter_name}::{flat_index}"
                        )

                        try:
                            row = next(iterator)
                        except StopIteration as exc:
                            raise ValueError(
                                f"Submission ended before {expected_row_id}"
                            ) from exc

                        if row["row_id"] != expected_row_id:
                            raise ValueError(
                                f"Unexpected row_id={row['row_id']!r}; "
                                f"expected={expected_row_id!r}"
                            )

                        try:
                            value = float(row["value"])
                        except Exception as exc:
                            raise ValueError(
                                f"Non-numeric value at {expected_row_id}"
                            ) from exc

                        if not math.isfinite(value):
                            raise ValueError(
                                f"Non-finite value at {expected_row_id}"
                            )

                        values.append(value)

                    state[parameter_name] = torch.tensor(
                        values,
                        dtype=torch.float32,
                    ).reshape(shape)

                state = validate_state_dict(
                    state,
                    (
                        f"case_{case_number}/"
                        f"malicious_{malicious_index}"
                    ),
                )

                model = SmallCNN()
                model.load_state_dict(state, strict=True)
                model.eval()

                with torch.inference_mode():
                    output = model(
                        torch.zeros(1, 3, 64, 64)
                    )

                if tuple(output.shape) != (1, 4):
                    raise ValueError(
                        "SmallCNN forward-pass output is invalid."
                    )

        try:
            extra = next(iterator)
        except StopIteration:
            extra = None

        if extra is not None:
            raise ValueError(
                f"Unexpected extra row: {extra.get('row_id')!r}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "attack_submission.csv",
    )
    args = parser.parse_args()

    try:
        validate_submission(args.submission)
    except Exception as exc:
        print("not valid")
        print(f"Reason: {exc}")
        raise SystemExit(1)

    print("valid")


if __name__ == "__main__":
    main()
