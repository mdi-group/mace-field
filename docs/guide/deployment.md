# Deployment and workflows

MACE-Field uses the MACE ASE calculator and command-line tools. The supported
field controls are described below; for general deployment options, see the
[MACE documentation](https://mace-docs.readthedocs.io/en/latest/).

## Selection and active-learning MD

Descriptor-based fine-tuning selection accepts a field model and field vector:

```bash
mace_finetuning_select \
  --configs_pt=pretraining.xyz \
  --configs_ft=field_train.xyz \
  --model=MACEField.model \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --output=selected.xyz
```

For committee active-learning MD, `--save_field_responses` writes the field
responses to the trajectory:

```bash
mace_active_learning_md \
  --config=initial.xyz \
  --model='MACEField_*.model' \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --save_field_responses \
  --output=trajectory.xyz
```

The saved polarization and polarizability are stored in `Atoms.info`;
per-atom BECs are stored in `Atoms.arrays`. For changing the field between
steps, use an ASE workflow that updates the calculator field.

## LAMMPS export

The LAMMPS exporter accepts one constant field vector for either supported
format:

```bash
mace_create_lammps_model MACEField.model \
  --format=libtorch --electric-field 0 0 0.02

mace_create_lammps_model MACEField.model \
  --format=mliap --electric-field 0 0 0.02
```

The field is stored in the exported wrapper; when omitted, the wrapper uses a
zero field. This interface does not transport a changing field from LAMMPS on
each timestep. Use ASE when the field changes during a trajectory.
