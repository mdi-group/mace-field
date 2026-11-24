HDF5 datasets
=============

For large datasets, ``mace_prepare_data`` writes sharded HDF5 files for
on-the-fly loading. A typical preparation command is:

.. code-block:: bash

   mace_prepare_data \
     --train_file=data/field_train.xyz \
     --valid_file=data/field_valid.xyz \
     --h5_prefix=processed/ \
     --r_max=5.0 \
     --num_process=8

The output is separated into ``train/``, ``val/``, and, when provided,
``test/`` directories. Point the training CLI at the relevant HDF5 file or
directory of shards:

.. code-block:: bash

   mace_run_train \
     --name=MACEField_hdf5 \
     --model=MACEField \
     --loss=universal_field \
     --train_file=processed/train \
     --valid_file=processed/val \
     --compute_polarization=True \
     --compute_becs=True \
     --compute_polarizability=True

The HDF5 paths preserve the field and response properties used by MACE-Field,
along with per-property weights. The reader supports the batched format written
by ``mace_prepare_data`` and the flat ``AtomicData`` format written by
``save_dataset_as_HDF5``.

.. note::

   In this checkout, the direct ``AtomicData`` HDF5 writer and reader include
   energy, forces, stress, virials, electric field, polarization, BECs, and
   polarizability. They do not serialize or load a Hessian target. A Hessian
   label therefore cannot be assumed to survive this HDF5 path.

For visualizing training metrics, select ``--error_table=PerAtomFieldRMSE`` to
include weighted polarization, BEC, and polarizability metrics in the training
tables and plots.
