# MACE-Field documentation

MACE-Field adds a uniform electric field to MACE and learns field-dependent
energies for molecules and periodic materials. Polarization and related response
properties are derivatives of the same energy used for forces.

Use this guide to install MACE-Field, prepare reference data, train or fine-tune
a model, and calculate field responses with ASE or the command-line tools.
For general MACE installation, training, and accelerator guidance, see the
[MACE documentation](https://mace-docs.readthedocs.io/en/latest/).

## Features

- Train the `MACEField` model with energy, force, stress, and electric-response
  labels.
- Obtain polarization, Born effective charges, and polarizability from
  derivatives of the field-dependent energy.
- Fine-tune a selected head from a MACE foundation model.
- Evaluate structures with the MACE CLI or ASE calculator and use a changing
  field in ASE workflows.
- Supervise the force Hessian with the `universal_field` loss.

```{toctree}
:maxdepth: 2
:caption: Quick Start

quick-start
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/index
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 1
:caption: Project

references
development
```

## Help

Report bugs and request features in the
[MACE-Field issue tracker](https://github.com/mdi-group/mace-field/issues).
