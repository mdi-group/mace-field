# Evaluation

Use `mace_eval_configs` to calculate model outputs for structures in an
ASE-readable file. Field-response output is opt-in for this command:

```bash
mace_eval_configs \
  --configs=input.xyz \
  --model=MACEField.model \
  --output=output.xyz \
  --compute_polarization \
  --compute_becs \
  --compute_polarizability
```

The field is read from the configured electric-field key in each structure.
Pass `--electric-field Ex Ey Ez` to apply one field vector to every
configuration, overriding the per-configuration input value. The field is in
V/Å.

By default, calculated values are written with the `MACE_` prefix:

| Output | Location |
| --- | --- |
| `MACE_polarization` | `Atoms.info` |
| `MACE_becs` | `Atoms.arrays` |
| `MACE_polarizability` | `Atoms.info` |

Use `--help` to see the available evaluation, batching, and output-key options.
For use within Python programs, see the [ASE calculator guide](ase-calculator.md).
