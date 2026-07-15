from collections import OrderedDict
from pathlib import Path
import re
import torch

from utilities.checks import (
    EXPECTED_STATE_LAYOUT,
    MALICIOUS_MODELS_PER_CASE,
    validate_state_dict,
)


def safe_torch_load(path):
    path = Path(path)

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_tensor_state_dict(value):
    return (
        isinstance(value, (dict, OrderedDict))
        and len(value) > 0
        and all(
            isinstance(key, str) and torch.is_tensor(tensor)
            for key, tensor in value.items()
        )
    )


def extract_state_dict(payload):
    if is_tensor_state_dict(payload):
        state = payload
    elif isinstance(payload, dict):
        state = None

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "weights",
            "net",
            "network",
        ):
            candidate = payload.get(key)

            if is_tensor_state_dict(candidate):
                state = candidate
                break

        if state is None:
            raise ValueError("No recognized state_dict was found.")
    else:
        raise TypeError(
            f"Unsupported model payload type: {type(payload).__name__}"
        )

    state = OrderedDict(state)

    for prefix in ("module.", "_orig_mod.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = OrderedDict(
                (key[len(prefix):], tensor)
                for key, tensor in state.items()
            )

    return validate_state_dict(state)


def load_state_dict(path):
    return extract_state_dict(safe_torch_load(path))


def save_state_dict(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(validate_state_dict(state), path)


def expected_client_paths(directory, expected_count):
    directory = Path(directory)

    return [
        directory / f"client_{index}.pt"
        for index in range(expected_count)
    ]


def load_state_dict_directory(directory, expected_count):
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"Missing model directory: {directory}")

    expected = expected_client_paths(directory, expected_count)
    missing = [str(path) for path in expected if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required client model files: " + ", ".join(missing)
        )

    actual = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    )

    if set(actual) != set(expected):
        extra = sorted(str(path) for path in set(actual) - set(expected))
        raise ValueError(
            f"Unexpected model files in {directory}: {extra}"
        )

    return [
        load_state_dict(path)
        for path in expected
    ]


def malicious_model_path(models_root, case_number, malicious_index):
    return (
        Path(models_root)
        / f"case_{case_number}"
        / f"malicious_{malicious_index}.pt"
    )


def iter_expected_attack_row_ids():
    for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
        for malicious_index in range(model_count):
            for parameter_name, shape in EXPECTED_STATE_LAYOUT.items():
                number_of_values = 1

                for dimension in shape:
                    number_of_values *= dimension

                for flat_index in range(number_of_values):
                    yield (
                        f"case_{case_number}::malicious_{malicious_index}::"
                        f"{parameter_name}::{flat_index}"
                    )
