# Contributing to TOPFARM

These notes describe the usual workflow for changes to TOPFARM.

## Getting Started

TOPFARM is developed on a DTU-hosted GitLab instance. External contributors
should use the public GitHub mirror:

1. Fork `https://github.com/DTUWindEnergy/TopFarm2`.
2. Clone your fork and install in developer mode:

   ```bash
   git clone https://github.com/<your-username>/TopFarm2.git
   cd TopFarm2
   pip install -e .
   pip install pre-commit
   pre-commit install
   ```

3. Create a branch from `master`.
4. Open a pull request against `DTUWindEnergy/TopFarm2`.

A DTU maintainer will review the PR on GitHub and apply accepted changes to
the GitLab repository.

## Branch Naming

| Type | Pattern | Example |
| --- | --- | --- |
| Bug fix | `fix/<short-description>` | `fix/mutable-default-args` |
| New feature | `feature/<short-description>` | `feature/hub-height-dv` |
| Refactor | `refactor/<short-description>` | `refactor/explicit-exports` |
| Documentation | `docs/<short-description>` | `docs/notebook-index` |
| Chore / cleanup | `chore/<short-description>` | `chore/remove-deprecated-code` |

## Code Style

- All code is auto-formatted with [autopep8](https://github.com/hhatto/autopep8) via pre-commit hooks. Run `pre-commit run --all-files` before pushing.
- Follow existing docstring style (NumPy format with `Parameters`, `Returns`, and `Notes` sections).
- Do not add comments that restate what the code does. Comment on why when the reason is non-obvious.

## Docstrings

Public classes and functions should use NumPy-style docstrings:

```python
def my_func(x, n_wt):
    """Short one-line summary.

    Parameters
    ----------
    x : array_like of shape (n_wt,)
        x-coordinates of turbines [m]
    n_wt : int
        Number of wind turbines

    Returns
    -------
    float
        Description of return value
    """
```

## Tests

- All new features and bug fixes must include tests.
- Tests live under `topfarm/tests/`. Mirror the module structure (e.g. tests for `topfarm/utils.py` go in `topfarm/tests/test_utils.py`).
- Run the test suite with:

  ```bash
  pytest topfarm/tests/
  ```

- Aim for 90% coverage on new code. Check with:

  ```bash
  pytest --cov=topfarm topfarm/tests/
  ```

## Reporting Issues

Please include:

- A minimal reproducible example
- TopFarm version (`python -c "import topfarm; print(topfarm.__version__)"`)
- Python and OpenMDAO versions
