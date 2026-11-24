Data and labels
===============

MACE-Field reads ASE configurations. Extended XYZ is a convenient text format;
HDF5 can be used for larger datasets. The default keys follow the MACE key
specification and can be changed with the corresponding ``*_key`` training
options.

Reference fields
----------------

Configuration-level values are stored in ``atoms.info``:

.. list-table:: Default configuration keys
   :header-rows: 1

   * - Key
     - Shape
     - Units
   * - ``REF_energy``
     - scalar
     - eV
   * - ``REF_stress``
     - ``(6,)`` or ``(3, 3)``
     - eV/Å³
   * - ``REF_virials``
     - ``(6,)`` or ``(3, 3)``
     - eV
   * - ``REF_electric_field``
     - ``(3,)``
     - V/Å
   * - ``REF_polarization``
     - ``(3,)``
     - e/Å²
   * - ``REF_polarizability``
     - ``(3, 3)`` or ``(9,)``
     - e/(V·Å)
   * - ``head``
     - string
     - multi-head name

Per-atom values are stored in ``atoms.arrays``:

.. list-table:: Default per-atom keys
   :header-rows: 1

   * - Key
     - Shape
     - Units
   * - ``REF_forces``
     - ``(N, 3)``
     - eV/Å
   * - ``REF_becs``
     - ``(N, 3, 3)`` or ``(N, 9)``
     - e

Here :math:`N` is the number of atoms. BECs are Born effective charge tensors,
with one tensor per atom. Check the source dataset's convention before
converting or relabelling response tensors.

Sparse and heterogeneous labels
-------------------------------

An individual configuration can omit properties it does not provide. The
loader assigns zero sample weight to an absent property. To deliberately mask
a property that is present, set its per-configuration weight in ``atoms.info``;
for example, ``config_polarization_weight = 0.0``. The corresponding global
loss weight controls the scale for that property across the dataset.

Field targets are needed only for nonzero field-response loss terms. The
electric field defaults to zero when it is absent. Keep fields and response
labels paired with the same structure and calculation conditions.

For periodic polarization, the default field loss folds the prediction against
the reference modulo the polarization lattice. This handles equivalent Berry
phase branches in the loss; it does not replace careful branch tracking when
building train and validation splits.
