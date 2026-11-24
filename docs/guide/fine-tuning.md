# Fine-tuning foundation models

MACE-Field can initialize from a compatible MACE foundation model. For a
multi-head checkpoint, select the pretrained head that matches the target data
domain. Consult the release notes for the checkpoint's supported elements,
reference method, and license before training.

The following example fine-tunes the `omat_pbe` head from MACE-MH-1 into a
single-head MACE-Field model:

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
