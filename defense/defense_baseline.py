from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")

    if len(models) != num_models:
        raise ValueError(
            f"num_models={num_models}, "
            f"but received {len(models)} models."
        )

    parameter_names = list(models[0].keys())
    result = OrderedDict()

    for name in parameter_names:
        reference = models[0][name]
        accumulator = torch.zeros_like(
            reference,
            dtype=torch.float64,
            device="cpu",
        )

        for model in models:
            value = model[name]

            if value.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name!r}."
                )

            accumulator.add_(
                value.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )

        result[name] = (
            accumulator
            .div(float(num_models))
            .to(dtype=reference.dtype)
            .contiguous()
        )

    return result
