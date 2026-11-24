<p align="center">
  <img src="macefield_logo.png" alt="MACE-Field logo" width="600">
</p>

# MACE-Field: Electric-Field-Aware Atomistic Models

[![Paper: PRX Intelligence](https://img.shields.io/badge/Paper-PRX%20Intelligence-blue)](https://doi.org/10.1103/b116-xy8k)
[![PyPI version](https://img.shields.io/pypi/v/mace-field-torch)](https://pypi.org/project/mace-field-torch/)
[![Core tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-core.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-core.yaml)
[![Extension tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-extensions.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-extensions.yaml)
[![Integration tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-integrations.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-integrations.yaml)
[![Documentation Status](https://app.readthedocs.org/projects/mace-field/badge/?version=latest)](https://mace-field.readthedocs.io/en/latest/?badge=latest)
[![Latest release](https://img.shields.io/github/v/release/mdi-group/mace-field)](https://github.com/mdi-group/mace-field/releases)

MACE-Field extends [MACE](https://github.com/ACEsuit/mace) with a uniform
electric-field input. It predicts a field-dependent scalar energy and derives
polarization, Born effective charges (BECs), and electric susceptibility from
that same scalar. The implementation supports training, fine-tuning, and
inference through the MACE command-line tools and ASE calculator.

## Documentation

Read the [MACE-Field user guide](https://mace-field.readthedocs.io/en/latest/)
for installation, data formats, training, fine-tuning, evaluation, and
calculator examples. The [MACE-Field tutorials repository](https://github.com/mdi-group/mace-field-tutorial)
contains MACE-Field-only notebooks, example data, and Colab launch links.

## Install

Install PyTorch for your hardware first with the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install MACE-Field from PyPI:

```bash
python -m pip install mace-field-torch
```

For CUDA setup and accelerator options, see the
[MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).
The distribution is named `mace-field-torch`; its Python import namespace is
`mace`. Avoid installing it in the same environment as another distribution
that also provides `mace`.

## MACE-Field foundation models

The model checkpoints are distributed as GitHub release assets, separately
from the PyPI package. The table lists the released MACE-Field foundations and
the data used for their field-response fine-tuning.

| Checkpoint | Base model and head | Fine-tuning data | Release |
| --- | --- | --- | --- |
| [`MACEField-MH-0-omat-dielectric.model`](https://github.com/mdi-group/mace-field/releases/tag/1.0.2) | MACE multi-head MP-0 (`mace_mh_1` release), `omat` head | Updated MP-Dielectric response data. The release also provides an MP-Dielectric/MP-Ferroelectric replay set for further fine-tuning. | [1.0.2](https://github.com/mdi-group/mace-field/releases/tag/1.0.2) |
| [`MACEField-MH-1.model`](https://github.com/mdi-group/mace-field/releases/tag/1.1.0) | MACE-MH-1, `omat_pbe` head | Curated MP-Ferroelectric (polarization), MP-Dielectric (BECs and polarizability), and Pheasy-MP-Phonon-v1.1 (Hessians, BECs, and polarizability). | [1.1.0](https://github.com/mdi-group/mace-field/releases/tag/1.1.0) |

For the MH-1 release, energy, force, and stress labels on these datasets are
zero-field MACE-MH-1 teacher predictions, not independent DFT references. Its
combined replay asset contains all 32,475 training and validation frames from
the three sources; this run has no independent test split. See the
[1.1.0 release](https://github.com/mdi-group/mace-field/releases/tag/1.1.0)
for the model, standalone replay set, complete run archive, and checksums. The
[quick start](https://mace-field.readthedocs.io/en/latest/quick-start.html)
shows checkpoint loading, and the [fine-tuning guide](https://mace-field.readthedocs.io/en/latest/guide/fine-tuning.html)
covers adapting MACE and MACE-Field foundations.

## Architecture

![MACE-Field architecture](macefield_architecture.png)

A single field-dependent energy supplies the mechanical and response
properties. For periodic structures, the model returns volume-normalized
polarization, volume-scaled Born effective charges, and dimensionless electric
susceptibility in vacuum-permittivity units. See the
[model conventions](https://mace-field.readthedocs.io/en/latest/guide/introduction.html)
and [data guide](https://mace-field.readthedocs.io/en/latest/guide/data.html)
for equations, units, label keys, and tensor shapes.

## Cite

If you use MACE-Field, cite the [PRX Intelligence paper](https://doi.org/10.1103/b116-xy8k)
and the MACE architecture papers listed in the
[references](https://mace-field.readthedocs.io/en/latest/references.html).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code checks, and
pull request guidance. Report bugs and request features in the
[issue tracker](https://github.com/mdi-group/mace-field/issues).
