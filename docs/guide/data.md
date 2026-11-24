# Data

Training data are ASE-readable configurations, usually stored as extended XYZ.
MACE-Field uses the standard MACE energy, force, stress, virial, and multi-head
keys, with additional keys for field responses. The defaults are:

| Quantity | Default key | Location | Shape | Units |
| --- | --- | --- | --- | --- |
| Energy | `REF_energy` | `Atoms.info` | scalar | eV |
| Forces | `REF_forces` | `Atoms.arrays` | `(N, 3)` | eV/Å |
| Stress | `REF_stress` | `Atoms.info` | `(6,)` or `(3, 3)` | eV/Å³ |
| Virials | `REF_virials` | `Atoms.info` | `(6,)` or `(3, 3)` | eV |
| Electric field | `REF_electric_field` | `Atoms.info` | `(3,)` | V/Å |
| Polarization | `REF_polarization` | `Atoms.info` | `(3,)` | e/Å² |
| Born effective charges | `REF_becs` | `Atoms.arrays` | `(N, 3, 3)` or `(N, 9)` | e |
| Electric susceptibility (`polarizability` API) | `REF_polarizability` | `Atoms.info` | `(3, 3)` or `(9,)` | dimensionless |
| Hessian | `REF_hessian` | `Atoms.info` | flattened `(3N, 3N)` | eV/Å² |
| Head | `head` | `Atoms.info` | string | — |

Key names can be changed with the corresponding `*_key` training options, for
example `--polarization_key=my_polarization`.

## Field-response labels

Provide `REF_electric_field` for each configuration whose label was computed
at a nonzero field. The training loss for each response quantity is controlled
by its own weight. Request the corresponding model output with
`--compute_polarization`, `--compute_becs`, or `--compute_polarizability`.
Computing BECs or polarizability also requires polarization internally.

If a field value is absent, the model input defaults to zero. Do not give a
nonzero weight to a response target that is missing from that configuration.
For heterogeneous datasets, use the per-configuration property weights to
mark which labels are present.

## Hessian labels

The optional Hessian target is the energy Hessian, equivalently the negative
position derivative of forces. Its default key is `REF_hessian`; its units are
eV/Å². For an `N`-atom structure, provide a `(3N, 3N)` matrix flattened into
`Atoms.info` for extended XYZ:

```python
atoms.info["REF_hessian"] = reference_hessian.reshape(-1)
```

The default Hessian loss weight is zero, so existing datasets do not request
Hessian calculations. Set `--hessian_weight` to a nonzero value to train on
these labels. Structures without a Hessian label have zero Hessian weight.
Mixed-size batches are supported. See [Training](training.md#force-hessian-supervision)
for a full CLI example.

## HDF5 data

The MACE data preparation and HDF5 interfaces preserve electric fields,
response tensors, per-property weights, and Hessians. See the MACE guide for
[large-dataset preprocessing](https://mace-docs.readthedocs.io/en/latest/guide/multipreprocessing.html)
and use the MACE-Field keys shown above when preparing field-labelled data.
