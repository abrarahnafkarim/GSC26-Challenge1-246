from pathlib import Path
import argparse
import ast
import importlib.util
import inspect
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SmallCNN
from utilities.checks import (
    VISIBLE_DEFENSE_MODEL_COUNT,
    validate_state_dict,
)
from utilities.model_io import load_state_dict_directory


FORBIDDEN_IMPORT_ROOTS = {
    "requests",
    "urllib",
    "http",
    "ftplib",
    "socket",
    "subprocess",
    "pathlib",
    "shutil",
    "pickle",
    "joblib",
}

FORBIDDEN_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
}


def static_restriction_check(path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]

                if root in FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(
                        "Forbidden import in defense submission: "
                        f"{alias.name}"
                    )

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]

            if root in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(
                    "Forbidden import in defense submission: "
                    f"{node.module}"
                )

        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALL_NAMES
            ):
                raise ValueError(
                    "Forbidden call in defense submission: "
                    f"{node.func.id}"
                )


def load_submission_function(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing defense submission: {path}. "
            "Copy defense/defense_submission_template.py to "
            "defense_submission.py first."
        )

    static_restriction_check(path)

    spec = importlib.util.spec_from_file_location(
        "participant_defense_submission",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, "robust_aggregation", None)

    if not callable(function):
        raise ValueError(
            "defense_submission.py must contain "
            "robust_aggregation."
        )

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())

    if len(parameters) != 2:
        raise ValueError(
            "robust_aggregation must accept exactly two arguments."
        )

    if [parameter.name for parameter in parameters] != [
        "num_models",
        "models",
    ]:
        raise ValueError(
            "The required argument order is: num_models, models."
        )

    return function


def clone_models(models):
    return [
        {
            key: tensor.detach().cpu().clone()
            for key, tensor in model.items()
        }
        for model in models
    ]


def assert_inputs_unchanged(before, after):
    if len(before) != len(after):
        raise ValueError("The submission modified the model list.")

    for model_index, (left, right) in enumerate(zip(before, after)):
        if list(left.keys()) != list(right.keys()):
            raise ValueError(
                f"Input model {model_index} keys were modified."
            )

        for key in left:
            if not torch.equal(left[key], right[key]):
                raise ValueError(
                    f"Input model {model_index}, parameter {key}, "
                    "was modified in place."
                )


def run_structural_tests(submission_path, visible_case_dir):
    function = load_submission_function(submission_path)

    all_models = load_state_dict_directory(
        visible_case_dir,
        expected_count=VISIBLE_DEFENSE_MODEL_COUNT,
    )

    test_counts = [5, VISIBLE_DEFENSE_MODEL_COUNT]

    for count in test_counts:
        supplied = clone_models(all_models[:count])
        before = clone_models(supplied)

        result = function(count, supplied)

        assert_inputs_unchanged(before, supplied)

        result = validate_state_dict(
            result,
            f"robust_aggregation output for {count} models",
        )

        model = SmallCNN()
        model.load_state_dict(result, strict=True)
        model.eval()

        with torch.inference_mode():
            output = model(
                torch.zeros(2, 3, 64, 64)
            )

        if tuple(output.shape) != (2, 4):
            raise ValueError(
                f"Invalid forward output shape: {tuple(output.shape)}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "defense_submission.py",
    )
    parser.add_argument(
        "--visible-case-dir",
        type=Path,
        default=ROOT / "defense" / "visible_case",
    )
    args = parser.parse_args()

    try:
        run_structural_tests(
            args.submission,
            args.visible_case_dir,
        )
    except Exception as exc:
        print("not valid")
        print(f"Reason: {exc}")
        raise SystemExit(1)

    print("valid")


if __name__ == "__main__":
    main()
