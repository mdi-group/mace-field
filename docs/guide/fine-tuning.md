# Fine-tuning from pretrained models

MACEField can start from either an upstream MACE pretrained model or an
existing MACEField checkpoint. These are different starting points:

- An upstream MACE model such as `mace-mh-1.model` supplies compatible MACE
  backbone and energy-readout parameters. The new MACEField model learns its
  field-dependent response from the response labels in your data.
- A MACEField checkpoint such as `MACEField-MH-1.model` or
  `MACEField-MH-0-omat-dielectric.model` can also supply its field-response
  parameters. Select a source head that exists in that checkpoint and matches
  your target data.

The released MACEField checkpoints, their upstream MACE starting models, and
their fine-tuning datasets are listed in the
[project README](https://github.com/mdi-group/mace-field#macefield-pretrained-foundation-models).
The latest checkpoint, `MACEField-MH-1.model`, is available in the
[1.1.0 release](https://github.com/mdi-group/mace-field/releases/tag/1.1.0).

## Start from an upstream MACE pretrained model

This example starts from the upstream MACE pretrained model
`mace-mh-1.model`, using its `omat_pbe` head, and creates a new MACEField
model:

```bash
mace_run_train \
  --name=MACEField_from_mace_mh_1_omat_pbe \
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

MACEField transfers compatible interaction, radial, product, and ordinary
energy-readout parameters from the selected MACE head. MACEField-specific
field-coupling parameters are trained against the available response labels.
Set `--foundation_model_elements=True` when the target model must retain all
elements in the pretrained MACE model.

For replay-based multi-head fine-tuning, set
`--multiheads_finetuning=True`, provide `--pt_train_file`, and define the
target head or heads with `--heads`. Select the pretrained source head with
`--foundation_head`. MACE-Field uses `universal_field` in this mode as well.

See the MACE documentation on [fine-tuning pretrained models](https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html)
and [multi-head replay](https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html)
for details on the upstream MACE options.

## Continue from a MACEField checkpoint

To continue training from a released MACEField checkpoint, pass its path to
`--foundation_model` and select one of its available heads with
`--foundation_head`. The example below uses the `mp-dielectric` head from
`MACEField-MH-0-omat-dielectric.model` (release 1.0.2). Set `--E0s` for the
target data; the energy baseline is configured from that data.

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
[MACEField 1.0.2 release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2).
For `MACEField-MH-1.model`, use a head present in that checkpoint. Set the loss
weights to match your data and objective. `--compute_*` enables model outputs;
a property contributes to training only when its labels and per-configuration
weights are present. For multi-head continuation, also provide the target head
mapping and replay data.
