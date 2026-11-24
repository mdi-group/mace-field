# Installation

Install PyTorch for your hardware first, following the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
Then install the released package from PyPI:

```bash
python -m pip install mace-field-torch
```

The current package release is 1.1.0. The distribution name is
`mace-field-torch`, while the Python import namespace remains `mace`. Avoid
installing it in the same environment as another distribution that provides
the `mace` namespace.

## Install from source

For development or to work with a checkout, install MACE-Field in editable
mode instead:

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

General MACE requirements, CUDA setup, and optional accelerator backends are
covered in the [MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html)
and [CUDA acceleration guide](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).
