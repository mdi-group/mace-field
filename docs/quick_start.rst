Quick Start
===========

Installation
------------

Clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/mdi-group/mace-field.git
   cd mace-field
   python -m pip install -e .

Install a PyTorch build suitable for your hardware before installing MACE-Field.
For the generic MACE requirements and CUDA guidance, see the
`MACE installation guide <https://mace-docs.readthedocs.io/en/latest/guide/installation.html>`_.

Download the released checkpoint
---------------------------------

The `1.0.2 release
<https://github.com/mdi-group/mace-field/releases/tag/1.0.2>`_ provides the
multi-head MH-0 checkpoint. Download
`MACEField-MH-0-omat-dielectric.model
<https://github.com/mdi-group/mace-field/releases/download/1.0.2/MACEField-MH-0-omat-dielectric.model>`_
and save it in your working directory.

Evaluate a structure with ASE
-----------------------------

Select a head explicitly for a multi-head checkpoint. This example applies a
constant field along :math:`z` and asks the calculator for the electric
response:

.. code-block:: python

   from ase.build import bulk
   from mace.calculators import MACECalculator

   atoms = bulk("Si", "diamond", a=5.43)
   atoms.calc = MACECalculator(
       model_paths=["MACEField-MH-0-omat-dielectric.model"],
       model_type="MACEField",
       head="mp-dielectric",
       electric_field=(0.0, 0.0, 0.01),  # V/Å
       compute_polarization=True,
       compute_becs=True,
       compute_polarizability=True,
       device="cpu",
   )

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   polarization = atoms.calc.results["polarization"]
   becs = atoms.calc.results["becs"]
   polarizability = atoms.calc.results["polarizability"]

The energy is in eV and forces are in eV/Å. The response arrays are available
in the calculator results; their shapes and units are listed in
:doc:`user_guide/data`.

Next steps
----------

* Prepare labels in :doc:`user_guide/data`.
* Train or fine-tune a model in :doc:`user_guide/training`.
* Use ASE, batch evaluation, or a constant-field LAMMPS wrapper in
  :doc:`user_guide/inference`.
