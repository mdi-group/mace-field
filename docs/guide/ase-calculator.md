# ASE Calculator

`MACECalculator` accepts `model_type="MACEField"` and an optional uniform
field. If `electric_field` is supplied to the calculator, that vector is used
for every calculation and overrides fields stored on the `Atoms` object.
Otherwise, the calculator checks `atoms.info["electric_field"]`, then
`atoms.info["REF_electric_field"]`, and uses zero if neither key is present.

```python
from ase.io import read
from mace.calculators import MACECalculator

atoms = read("structure.xyz")
calc = MACECalculator(
    model_paths=["MACEField.model"],
    model_type="MACEField",
    electric_field=[0.0, 0.0, 0.02],  # V/Å
    compute_polarization=True,
    compute_becs=True,
    compute_polarizability=True,
    device="cuda",
)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = calc.results["polarization"]
becs = calc.results["becs"]
polarizability = calc.results["polarizability"]
```

Response flags can be selected independently. BECs and polarizability require
polarization internally. For a MACE-Field model, unspecified response flags
retain the historical default of calculating all three response outputs; set
unused flags to `False` to skip their derivatives. The returned response
tensors are:

- `polarization`: one three-vector per structure.
- `becs`: one `3 × 3` tensor per atom.
- `polarizability`: one `3 × 3` tensor per structure, reported as the
  dimensionless relative electric susceptibility in `ε₀` units.

## Changing the field

For a field that changes during an ASE workflow, update the calculator field
before each calculation:

```python
for field_z in field_schedule:
    atoms.calc.electric_field = [0.0, 0.0, field_z]
    energy = atoms.get_potential_energy()
```

The calculator field takes precedence over a field stored in `atoms.info`.
This makes field-dependent ASE relaxation and molecular-dynamics workflows
possible. The physical validity of a field path depends on the training data
and should be assessed separately.

## Hessian

For supported MACE-Field models, the calculator can also return the energy
Hessian:

```python
hessian = atoms.calc.get_hessian(atoms=atoms)
```

The result has shape `(3N, N, 3)`, which can be reshaped to `(3N, 3N)` for
matrix operations. It is the negative position derivative of forces.
