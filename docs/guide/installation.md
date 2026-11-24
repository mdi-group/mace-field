# Installation

Install PyTorch for your hardware first, following the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
The package release is pending. For now, install MACE-Field from a source
checkout in editable mode:

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

This checkout provides the `mace` Python import namespace. Avoid installing it
in the same environment as another distribution that also provides `mace`.

General MACE requirements, CUDA setup, and optional accelerator backends are
covered in the [MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html)
and [CUDA acceleration guide](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).
