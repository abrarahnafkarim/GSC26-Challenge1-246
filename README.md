# GSC 2026 Federated Learning Backdoor Attack and Defense Challenge

Read `participant_guide.pdf` and begin with `quickstart.ipynb`.

## Public starter structure

```text
challenge_starter/
├── README.md
├── participant_guide.pdf
├── model.py
├── quickstart.ipynb
├── attack/
│   ├── case_1/
│   │   ├── client_0.pt
│   │   ├── ...
│   │   └── client_7.pt
│   ├── case_2/
│   │   ├── client_0.pt
│   │   ├── ...
│   │   └── client_19.pt
│   ├── case_3/
│   │   ├── client_0.pt
│   │   ├── ...
│   │   └── client_14.pt
│   ├── sample_submission.csv
│   ├── attack_baseline.py
│   ├── create_attack_submission.py
│   └── validate_attack_submission.py
├── defense/
│   ├── visible_case/
│   │   ├── client_0.pt
│   │   ├── ...
│   │   └── client_9.pt
│   ├── defense_submission_template.py
│   ├── defense_baseline.py
│   └── test_defense_submission.py
└── utilities/
    ├── __init__.py
    ├── model_io.py
    └── checks.py
```

Every supplied client model file contains one complete SmallCNN `state_dict`.
The visible defense model filenames and ordering do not identify model roles.

## Attack workflow

1. Inspect the supplied benign models in `attack/case_1`, `case_2`, and
   `case_3`.
2. Run `attack/attack_baseline.py` or create your own malicious models.
3. Save your malicious models as:

```text
participant_models/
├── case_1/
│   ├── malicious_0.pt
│   └── malicious_1.pt
├── case_2/
│   ├── malicious_0.pt
│   ├── malicious_1.pt
│   ├── malicious_2.pt
│   ├── malicious_3.pt
│   └── malicious_4.pt
└── case_3/
    ├── malicious_0.pt
    ├── malicious_1.pt
    ├── malicious_2.pt
    ├── malicious_3.pt
    └── malicious_4.pt
```

4. Run:

```bash
python attack/create_attack_submission.py
python attack/validate_attack_submission.py
```

5. Confirm that the validator prints `valid`.

The final attack file must be named `attack_submission.csv`.

## Defense workflow

1. Copy `defense/defense_submission_template.py` to
   `defense_submission.py`.
2. Implement:

```python
def robust_aggregation(num_models, models):
    ...
```

3. Run:

```bash
python defense/test_defense_submission.py
```

4. Confirm that the validator prints `valid`.

The final defense file must be named `defense_submission.py`.

## Final portal upload

Upload both:

- `attack_submission.csv`
- `defense_submission.py`

Do not upload the client model `.pt` files to the submission portal.
