Training and fine-tuning
========================

MACE-Field uses the standard MACE training CLI with ``--model=MACEField`` and
``--loss=universal_field``. See the
`MACE Training guide <https://mace-docs.readthedocs.io/en/latest/guide/training.html>`_
for general model and optimizer options.

Training from scratch
---------------------

The following is a starting command for a dataset with mechanical and field
response labels:

.. code-block:: bash

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

The sample weights and loss weights are separate. The sample weight determines
whether a configuration contributes a target; the global loss weight sets the
relative strength of that target's loss term. Choose loss weights for the scale
and scientific priority of the task rather than copying the illustrative values
without checking them.

Fine-tuning MACE-MH-1
---------------------

MACE-MH-1 is a multi-head foundation model. Download it from the
`MACE foundation-model release
<https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1>`_ and
choose the foundation head that matches the target chemistry. The example below
transfers one head into a one-head MACE-Field model:

.. code-block:: bash

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

Compatible interaction and energy-readout parameters are transferred from the
foundation model. ``--E0s=estimated`` estimates reference-energy shifts from
the fine-tuning data using the foundation model predictions. For general
fine-tuning decisions and caveats, see the
`MACE fine-tuning guide <https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html>`_.

For replay-based multi-head fine-tuning, set
``--multiheads_finetuning=True``, provide a replay dataset with
``--pt_train_file``, and define the target heads with ``--heads``. Select the
foundation head with ``--foundation_head``.

Replay pseudolabels
-------------------

``--pseudolabel_replay`` generates mechanical replay labels from the source
model. A MACEField source can additionally provide response pseudolabels when
the corresponding ``--compute_polarization``, ``--compute_becs``, and
``--compute_polarizability`` options are enabled. Requesting BECs or
polarizability also requires polarization.

A plain MACE foundation checkpoint can provide energy, forces, and stress; it
cannot generate polarization, BEC, or polarizability labels. Use a MACEField
checkpoint as the replay source when response pseudolabels are needed. Replay
does not create response supervision from a source model that has no field
response.
