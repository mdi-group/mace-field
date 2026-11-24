# Quick Start

This page takes you from a source checkout to a first MACE-Field training
command. It assumes an ASE-readable extended XYZ dataset with reference energy
and force labels. See [Data](guide/data.md) for the optional field-response
labels.

## Install

Install a PyTorch build that matches your CPU or CUDA environment. After the
first PyPI release, install MACE-Field with:

```bash
python -m pip install mace-field-torch
```

To use the current development version, install this checkout:

```bash
git clone https://github.com/mdi-group/mace-field.git
cd mace-field
python -m pip install -e .
```

For the editable installation and development tools, see
[Installation](guide/installation.md).

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
