"""Pre-submission checker for defense_submission.py.

Run this BEFORE every defense upload. It catches the two ways a defense gets
rejected that the official test_defense_submission.py does NOT catch:

  1. SUBMISSION GUARD: the evaluator's preflight allows only imports and the
     robust_aggregation function at module scope. Module-level constants,
     assignments, or helper defs cause:
       "only imports and robust_aggregation are allowed at module scope"

  2. COMPLIANCE: the organizers forbid handcrafted knowledge of the challenge -
     no class-specific rules, no targeting/suppressing a particular class, no
     external data or pretrained models. Everything must be derived only from
     the client models passed in.

Usage:
    python solution/check_defense_submission.py
    python solution/check_defense_submission.py --submission path/to/file.py
"""

from pathlib import Path
import argparse
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

# Words that suggest hardcoded challenge knowledge (checked in CODE, not docs).
SUSPICIOUS = [
    "target_class", "target_index", "black", "blond", "brown", "gray",
    "sunglass", "mask", "trigger", "backdoor_class",
]


def check_module_scope(tree):
    """Only a module docstring, imports, and robust_aggregation are allowed."""
    violations = []
    for i, node in enumerate(tree.body):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name != "robust_aggregation":
                violations.append(
                    f"line {line}: module-scope helper def '{node.name}' "
                    f"(move it INSIDE robust_aggregation)"
                )
            continue
        if (i == 0 and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue  # module docstring
        violations.append(
            f"line {line}: {type(node).__name__} at module scope "
            f"(only imports and robust_aggregation are allowed)"
        )
    return violations


def check_compliance(src, tree):
    """Flag hardcoded class/challenge knowledge in CODE (ignores comments/docs)."""
    findings = []

    # Strip docstrings so prose doesn't produce false positives.
    code_lines = src.splitlines()
    doc_spans = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_spans.add(ln)

    for idx, line in enumerate(code_lines, start=1):
        if idx in doc_spans:
            continue
        stripped = line.split("#", 1)[0]        # drop inline comments
        low = stripped.lower()
        for word in SUSPICIOUS:
            if word in low:
                findings.append(f"line {idx}: mentions '{word}' -> {stripped.strip()}")

    # Indexing a specific class of the classifier is the classic violation.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            seg = ast.get_source_segment(src, node) or ""
            if "classifier" in seg and re.search(r"\[\s*\d+\s*[,\]]", seg):
                findings.append(
                    f"line {node.lineno}: class-specific classifier indexing -> {seg}"
                )
    return findings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", type=Path,
                   default=ROOT / "defense_submission.py")
    args = p.parse_args()

    src = args.submission.read_text()
    tree = ast.parse(src)

    guard = check_module_scope(tree)
    comp = check_compliance(src, tree)

    print(f"Checking: {args.submission}\n")
    print("[1] Submission guard (module scope)")
    if guard:
        for v in guard:
            print("  FAIL -", v)
    else:
        print("  PASS - only imports + robust_aggregation at module scope")

    print("\n[2] Compliance (no handcrafted challenge knowledge)")
    if comp:
        for v in comp:
            print("  WARN -", v)
        print("  -> Review each: class-specific rules are DISQUALIFYING.")
    else:
        print("  PASS - no class-specific or challenge-specific logic detected")

    if guard:
        print("\nRESULT: would be REJECTED at preflight.")
        raise SystemExit(1)
    print("\nRESULT: ready to submit (also run defense/test_defense_submission.py).")


if __name__ == "__main__":
    main()
