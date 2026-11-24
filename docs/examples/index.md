# Tutorials and examples

The notebooks, example datasets, and Google Colab launchers are maintained in
the separate [MACE-Field tutorials repository](https://github.com/mdi-group/mace-field-tutorial).

## Tutorial sequence

1. [MACEField architecture and response derivatives](https://github.com/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/01_macefield_architecture.ipynb) — [Open in Colab](https://colab.research.google.com/github/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/01_macefield_architecture.ipynb).
2. [Train on BaTiO₃ reference data](https://github.com/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/02_train_macefield_batio3.ipynb) — [Open in Colab](https://colab.research.google.com/github/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/02_train_macefield_batio3.ipynb).
3. [Finite-field ASE molecular dynamics](https://github.com/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/03_finite_field_mlmd_hysteresis.ipynb) — [Open in Colab](https://colab.research.google.com/github/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/03_finite_field_mlmd_hysteresis.ipynb).
4. [Explore the existing dielectric and ferroelectric datasets](https://github.com/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/04_response_datasets.ipynb) — [Open in Colab](https://colab.research.google.com/github/mdi-group/mace-field-tutorial/blob/main/tutorials/macefield/04_response_datasets.ipynb).

The course notebook setup clones this package's `main` branch and installs it
in editable mode. The BaTiO₃ reference set contains 59 135-atom structures and
a 48/11 train/validation split at one small positive field vector. The
finite-field trajectory is a software workflow demonstration, not a validated
hysteresis or coercive-field prediction. See the [tutorial repository](https://github.com/mdi-group/mace-field-tutorial)
for the response-dataset label map, split sizes, provenance notes, and coverage limits.

For broader MACE training and architecture material, see the [MACE tutorial index](https://mace-docs.readthedocs.io/en/latest/examples/tutorials.html).
