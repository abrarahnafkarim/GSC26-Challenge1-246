from collections import OrderedDict
import torch


EXPECTED_STATE_LAYOUT = OrderedDict([
    ("features.0.weight", (32, 3, 3, 3)),
    ("features.0.bias", (32,)),
    ("features.3.weight", (64, 32, 3, 3)),
    ("features.3.bias", (64,)),
    ("features.6.weight", (128, 64, 3, 3)),
    ("features.6.bias", (128,)),
    ("classifier.weight", (4, 128)),
    ("classifier.bias", (4,)),
])

EXPECTED_PARAMETER_COUNT = 93_764

ATTACK_BENIGN_MODELS_PER_CASE = OrderedDict([
    (1, 8),
    (2, 20),
    (3, 15),
])

MALICIOUS_MODELS_PER_CASE = OrderedDict([
    (1, 2),
    (2, 5),
    (3, 5),
])

VISIBLE_DEFENSE_MODEL_COUNT = 10


def validate_state_dict(state, description="state_dict"):
    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(f"{description} must be a mapping.")

    if set(state) != set(EXPECTED_STATE_LAYOUT):
        missing = sorted(set(EXPECTED_STATE_LAYOUT) - set(state))
        extra = sorted(set(state) - set(EXPECTED_STATE_LAYOUT))
        raise ValueError(
            f"{description} has incorrect parameter keys. "
            f"Missing={missing}, extra={extra}"
        )

    normalized = OrderedDict()
    total = 0

    for key, expected_shape in EXPECTED_STATE_LAYOUT.items():
        tensor = state[key]

        if not torch.is_tensor(tensor):
            raise TypeError(f"{description}[{key!r}] is not a tensor.")

        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{description}[{key!r}] has shape {tuple(tensor.shape)}; "
                f"expected {expected_shape}."
            )

        value = tensor.detach().cpu().to(torch.float32).contiguous()

        if not torch.isfinite(value).all().item():
            raise ValueError(
                f"{description}[{key!r}] contains NaN or infinity."
            )

        normalized[key] = value.clone()
        total += value.numel()

    if total != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{description} contains {total:,} values; "
            f"expected {EXPECTED_PARAMETER_COUNT:,}."
        )

    return normalized


def validate_collection(models, expected_count=None, description="models"):
    if not isinstance(models, (list, tuple)):
        raise TypeError(f"{description} must be a list or tuple.")

    if expected_count is not None and len(models) != expected_count:
        raise ValueError(
            f"{description} contains {len(models)} models; "
            f"expected {expected_count}."
        )

    if not models:
        raise ValueError(f"{description} is empty.")

    return [
        validate_state_dict(model, f"{description}[{index}]")
        for index, model in enumerate(models)
    ]


def fedavg(models):
    validated = validate_collection(models)
    aggregated = OrderedDict()

    for key in EXPECTED_STATE_LAYOUT:
        accumulator = torch.zeros_like(
            validated[0][key],
            dtype=torch.float64,
            device="cpu",
        )

        for state in validated:
            accumulator.add_(state[key].to(torch.float64))

        aggregated[key] = (
            accumulator
            .div(float(len(validated)))
            .to(torch.float32)
            .contiguous()
        )

    return validate_state_dict(aggregated, "FedAvg output")


def expected_attack_row_count():
    return EXPECTED_PARAMETER_COUNT * sum(MALICIOUS_MODELS_PER_CASE.values())
