# Installation

Install PyTorch for your hardware first, following the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
After the first PyPI release, install MACE-Field with:

```bash
python -m pip install mace-field-torch
```

For the current development version, install the repository in editable mode:

```bash
git clone https://github.com/mdi-group/mace-field.git
cd mace-field
python -m pip install -e .
```

The editable install provides the `mace_run_train`, `mace_eval_configs`, and
other MACE command-line entry points. To install development dependencies and
the repository hooks, use:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

The `mace-field-torch` distribution uses the existing Python import name
`mace`. Do not install it in the same environment as `mace-torch`, since both
distributions provide that package.

General MACE requirements, CUDA setup, and optional accelerator backends are
covered in the [MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html)
and [CUDA acceleration guide](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).
