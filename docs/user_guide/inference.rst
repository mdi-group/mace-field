Inference
=========

ASE calculator
--------------

Use ``MACECalculator`` with ``model_type="MACEField"``. The field can be passed
to the calculator or stored per configuration. An explicit calculator field
overrides fields stored in ``atoms.info``; otherwise the calculator checks
``electric_field`` and then ``REF_electric_field`` before using zero field.

.. code-block:: python

   from ase.io import read
   from mace.calculators import MACECalculator

   atoms = read("structure.extxyz")
   atoms.calc = MACECalculator(
       model_paths=["MACEField.model"],
       model_type="MACEField",
       head="mp-dielectric",
       electric_field=(0.0, 0.0, 0.02),  # V/Å
       compute_polarization=True,
       compute_becs=True,
       compute_polarizability=True,
       device="cuda",
   )

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   polarization = atoms.calc.results["polarization"]
   becs = atoms.calc.results["becs"]
   polarizability = atoms.calc.results["polarizability"]

The calculator returns polarization with shape ``(3,)``, BECs with shape
``(N, 9)``, and polarizability with shape ``(9,)``. Reshape the latter two to
``(N, 3, 3)`` and ``(3, 3)`` when a matrix representation is more convenient.
For MACE-Field, the calculator computes the field-response outputs by default
when no response-selection flags are supplied. Set the flags explicitly to
avoid calculating unused derivatives. BEC and polarizability calculations also
require polarization.

Batch evaluation
----------------

Use ``mace_eval_configs`` to annotate an ASE-readable input file:

.. code-block:: bash

   mace_eval_configs \
     --configs=input.xyz \
     --model=MACEField.model \
     --output=output.xyz \
     --compute_polarization \
     --compute_becs \
     --compute_polarizability

Pass ``--electric-field 0 0 0.02`` to apply one field to every configuration.
By default, response results use the ``MACE_`` prefix: polarization and
polarizability are written to ``atoms.info`` and BECs to ``atoms.arrays``.

Changing the field during a simulation
---------------------------------------

ASE workflows can change the field between calculations:

.. code-block:: python

   atoms.calc.electric_field = (0.0, 0.0, Ez_t)

This supports finite-field relaxation, molecular dynamics, and response curves.
The LAMMPS wrappers store a constant field in the exported model; they do not
transport a changing per-step field. Use ASE when the field must change during
the simulation.
