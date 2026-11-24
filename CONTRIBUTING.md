# Contributing to MACE-Field

Thanks for helping improve MACE-Field. Contributions may include model and
calculator changes, bug reports, documentation, examples, and reproducibility
improvements.

## Report a problem or propose a change

Use the [issue tracker](https://github.com/mdi-group/mace-field/issues) for
bugs, questions, and feature proposals. For a bug, include a minimal example,
the expected and observed behavior, the MACE-Field version, Python and PyTorch
versions, operating system, device (CPU/CUDA), and a traceback with private
paths or data removed. Do not attach model checkpoints, large datasets, or
training logs unless a maintainer asks for a small, relevant excerpt.

For a change, open a pull request against `main` and describe the problem it
solves, the approach, and any impact on existing MACE or MACE-Field behavior.
For broad changes, open an issue first so the design can be discussed. Changes
to the shared MACE core should be proposed upstream when they are useful
without the field-response extension.

## Set up a development environment

Use Python 3.10 or newer. Install a PyTorch build suited to your hardware using
the [PyTorch selector](https://pytorch.org/get-started/locally/), then from a
checkout run:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

The `mace-field-torch` distribution and upstream `mace-torch` both install the
`mace` Python package. Use a clean environment and do not install both.

## Check your change

Run the relevant tests before opening a pull request. The CPU MACE-Field tests
are:

```bash
python -m pytest tests/extensions/macefield
```

For changes that affect shared data, models, or CLI behavior, also run the
relevant suites in `tests/unit` and `tests/workflows`. The GitHub Actions
workflows run lint, core CPU tests, and capability-specific extension tests.
Run the repository format and lint hooks with:

```bash
pre-commit run --all-files
```

The project release line follows the existing MACE-Field tags. The next planned
version is `1.1.0`; release tags must match `mace.__version__` exactly.

Update the user guide or examples when a change affects user-facing behavior.
Keep fixtures small and deterministic; do not commit training outputs, model
weights, or private or licensed datasets.

## Pull request checklist

- [ ] The change has a focused description and links to its issue when relevant.
- [ ] Tests cover changed behavior, or the pull request explains why they do
  not apply.
- [ ] User-facing documentation is updated where needed.
- [ ] No private data, generated model files, or unnecessary large artifacts are
  included.
- [ ] Any change to package behavior remains compatible with the documented
  `mace` import name and the `MACEField` model type.

## Support

Use the issue tracker for support requests and bug reports. For security issues,
do not post exploit details publicly; contact the repository maintainers through
the contact address in the README.
