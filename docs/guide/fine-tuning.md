# Fine-tuning foundation models

MACE-Field can initialize from a compatible MACE foundation model. For a
multi-head checkpoint, select the pretrained head that matches the target data
domain. Consult the release notes for the checkpoint's supported elements,
reference method, and license before training.

The released MACE-Field foundation checkpoints and their fine-tuning data are
listed in the
[project README](https://github.com/mdi-group/mace-field#mace-field-foundation-models).
The current MH-1 model is available from the
[1.1.0 release](https://github.com/mdi-group/mace-field/releases/tag/1.1.0).

The following example starts from the upstream MACE-MH-1 `omat_pbe` head and
fine-tunes it into a single-head MACE-Field model. It does not start from the
released MACEField-MH-1 checkpoint:

```bash
mace_run_train \
  --name=MACEField-MH-1-omat \
  --foundation_model=/path/to/mace-mh-1.model \
  --foundation_head=omat_pbe \
  --foundation_model_elements=False \
  --foundation_model_readout=True \
  --multiheads_finetuning=False \
  --model=MACEField \
  --loss=universal_field \
  --train_file=data/field_train.xyz \
  --valid_file=data/field_valid.xyz \
  --E0s=estimated \
  --compute_forces=True \
  --compute_stress=True \
  --compute_polarization=True \
  --compute_becs=True \
  --compute_polarizability=True \
  --device=cuda
```

MACE-Field transfers compatible interaction, radial, product, and ordinary
energy-readout parameters from the selected MACE head. The field-coupling
parameters are specific to MACE-Field and are trained against the available
response labels. Set `--foundation_model_elements=True` when the target model
must retain all elements in the foundation model.

For replay-based multi-head fine-tuning, set
`--multiheads_finetuning=True`, provide `--pt_train_file`, and define the
target head or heads with `--heads`. Select the pretrained source head with
`--foundation_head`. MACE-Field uses `universal_field` in this mode as well.

See the MACE documentation on [fine-tuning foundation models](https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html)
and [multi-head replay](https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html)
for general guidance.

## Fine-tuning from the MH-0 checkpoint

A MACE-Field checkpoint can initialize another MACE-Field model. This transfers
compatible backbone, energy-readout, and field-coupling parameters from the
selected source head. Choose `--foundation_head` to match the target domain;
for example, the released MH-0 checkpoint includes an `mp-dielectric` head.
The target energy baseline is configured from the fine-tuning data, so set
`--E0s` deliberately rather than assuming the source energy shift is reused.

```bash
mace_run_train \
  --name=MACEField-MH-0-finetuned \
  --foundation_model=/path/to/MACEField-MH-0-omat-dielectric.model \
  --foundation_head=mp-dielectric \
  --foundation_model_elements=False \
  --foundation_model_readout=True \
  --multiheads_finetuning=False \
  --model=MACEField \
  --loss=universal_field \
  --train_file=data/field_train.xyz \
  --valid_file=data/field_valid.xyz \
  --E0s=estimated \
  --compute_forces=True \
  --compute_stress=True \
  --compute_polarization=True \
  --compute_becs=True \
  --compute_polarizability=True \
  --energy_weight=1.0 \
  --polarization_weight=10.0 \
  --becs_weight=100.0 \
  --polarizability_weight=1.0 \
  --device=cuda
```

The example checkpoint is available from the
[MACE-Field 1.0.2 release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2).
Replace the selected head and loss weights to match the checkpoint and labels
you are using. `--compute_*` controls which predictions are evaluated; a
property contributes to training only when its labels and per-configuration
weights are present. For multi-head continuation, provide the head mapping and
replay data as described above, and ensure the source checkpoint contains the
selected head.
