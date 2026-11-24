# Training

Use `MACEField` with the `universal_field` loss. The usual MACE training
options control architecture, data splits, optimization, device selection, and
energy, force, and stress terms. MACE-Field adds response outputs and their
loss weights.

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
  --compute_stress=True \
  --compute_polarization=True \
  --compute_becs=True \
  --compute_polarizability=True \
  --energy_weight=1.0 \
  --forces_weight=100.0 \
  --stress_weight=1.0 \
  --polarization_weight=1.0 \
  --becs_weight=100.0 \
  --polarizability_weight=100.0 \
  --device=cuda
```

Use `python -m mace.cli.run_train` with the same options when running from a
source checkout without the installed entry point. The weights above are an
example only; scale them for the units, label coverage, and objective of your
dataset. See the MACE [training guide](https://mace-docs.readthedocs.io/en/latest/guide/training.html)
for general model and optimizer options.

## Periodic polarization

For periodic data, `universal_field` accounts for equivalent polarization
branches using the lattice. Keep the field, cell, and polarization convention
consistent between the reference calculations and training data. This branch
handling does not correct inconsistent labels or identify a physical switching
path from data that do not sample one.

## Pseudolabel replay

The `--pseudolabel_replay` option can generate energy and force labels from a
foundation model and can include stress/virial labels according to the replay
options. Field-response pseudolabels require a MACE-Field source model; a plain
MACE foundation model has no field-response head and cannot provide
polarization, BEC, or polarizability labels. When replaying from a field model,
request the needed response outputs with the corresponding `--compute_*`
options.

## Force Hessian supervision

To include the force Hessian in the `universal_field` objective, store a
flattened `REF_hessian` matrix for each labelled structure and set a nonzero
Hessian weight:

```bash
mace_run_train \
  --name=MACEField_hessian \
  --model=MACEField \
  --loss=universal_field \
  --train_file=data/field_train_with_hessians.xyz \
  --hessian_key=REF_hessian \
  --hessian_weight=1.0 \
  --device=cuda
```

The Hessian is $\partial^2 E/\partial\mathbf{R}^2=-\partial\mathbf{F}/\partial\mathbf{R}$,
with units eV/Å². The default Hessian weight is zero.
Configurations without Hessian labels can remain in the dataset with zero
Hessian weight. Batches can contain structures with different atom counts.

Hessian calculations involve second derivatives and can use substantially more
memory and time than energy/force training. Start with a small batch and
confirm the available memory before increasing it.
