<p align="center">
  <img src="macefield_logo.png" alt="MACE-Field logo" width="600">
</p>

# MACE-Field: Electric-Field-Aware Atomistic Models

[![Paper: PRX Intelligence](https://img.shields.io/badge/Paper-PRX%20Intelligence-blue)](https://doi.org/10.1103/b116-xy8k)
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

## Install from source

Package publication is pending; for now, install from a source checkout. Follow
the [MACE-Field installation guide](https://mace-field.readthedocs.io/en/latest/guide/installation.html)
and choose a compatible PyTorch build with the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
For CUDA setup, see the [MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).

## Released checkpoint

The [1.0.2 release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2)
includes an MH-0 checkpoint, a fine-tuning script, and a
dielectric/ferroelectric replay dataset. The [quick start](https://mace-field.readthedocs.io/en/latest/quick-start.html)
shows how to load the checkpoint; the [fine-tuning guide](https://mace-field.readthedocs.io/en/latest/guide/fine-tuning.html)
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
