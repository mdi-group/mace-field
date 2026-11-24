# Quick Start

This page takes you from installing MACE-Field to a first training command. It
assumes an ASE-readable extended XYZ dataset with reference energy and force
labels. See [Data](guide/data.md) for the optional field-response labels.

## Install

Install the PyPI package using the
[installation guide](guide/installation.md), including the PyTorch build for
your hardware. Return here for the first training and inference examples.

## Train a model

Start with a dataset called `field_train.xyz`:

```bash
mace_run_train \
  --name=MACEField_model \
  --model=MACEField \
  --loss=universal_field \
  --train_file=data/field_train.xyz \
  --valid_fraction=0.2 \
  --r_max=5.0 \
  --num_interactions=2 \
  --num_channels=128 \
  --max_L=1 \
  --compute_forces=True \
  --compute_polarization=True \
  --energy_weight=1.0 \
  --forces_weight=100.0 \
  --polarization_weight=1.0 \
  --device=cuda
```

Choose `--device=cpu` when CUDA is unavailable. Add the corresponding
`--compute_*` options and loss weights when the data contain stress, Born
effective charges, or polarizability labels. Periodic polarization differences
are folded over the lattice by the default `universal_field` loss.

## Calculate responses

Load the trained model with ASE and set a field in volts per ångström:

```python
from ase.io import read
from mace.calculators import MACECalculator

atoms = read("structure.xyz")
atoms.calc = MACECalculator(
    model_paths=["MACEField_model.model"],
    model_type="MACEField",
    electric_field=[0.0, 0.0, 0.02],
    compute_polarization=True,
    compute_becs=True,
    compute_polarizability=True,
    device="cuda",
)

energy = atoms.get_potential_energy()
polarization = atoms.calc.results["polarization"]
```

Continue with the [ASE calculator guide](guide/ase-calculator.md), or use the
[tutorial notebooks](examples/index.md) for a worked BaTiO₃ workflow.

## Try the MACEField-MH-0 checkpoint

This example uses the earlier MACEField checkpoint
`MACEField-MH-0-omat-dielectric.model`. The current checkpoint,
`MACEField-MH-1.model`, and the training-data provenance for both models are
listed in the
[foundation-model table in the README](https://github.com/mdi-group/mace-field#macefield-pretrained-foundation-models).
The [1.0.2 release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2)
includes the multi-head checkpoint
[`MACEField-MH-0-omat-dielectric.model`](https://github.com/mdi-group/mace-field/releases/download/1.0.2/MACEField-MH-0-omat-dielectric.model).
Download it and select the head that matches your data domain:

```python
from ase.build import bulk
from mace.calculators import MACECalculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = MACECalculator(
    model_paths=["MACEField-MH-0-omat-dielectric.model"],
    model_type="MACEField",
    head="mp-dielectric",
    electric_field=(0.0, 0.0, 0.01),  # V/Å
    compute_polarization=True,
    compute_becs=True,
    compute_polarizability=True,
    device="cpu",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = atoms.calc.results["polarization"]
```

See the [ASE calculator guide](guide/ase-calculator.md) for field selection
and additional outputs.
