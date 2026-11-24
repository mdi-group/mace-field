<p align="center">
  <img src="macefield_logo.png" alt="MACE-Field logo" width="600">
</p>

# MACE-Field: Electric-Field-Aware MACE Models

[![Core tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-core.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-core.yaml)
[![Extension tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-extensions.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-extensions.yaml)
[![Integration tests](https://github.com/mdi-group/mace-field/actions/workflows/ci-integrations.yaml/badge.svg?branch=main)](https://github.com/mdi-group/mace-field/actions/workflows/ci-integrations.yaml)
[![Latest release](https://img.shields.io/github/v/release/mdi-group/mace-field)](https://github.com/mdi-group/mace-field/releases)

MACE-Field extends the [MACE](https://github.com/ACEsuit/mace) architecture with a
uniform electric-field input. It learns a scalar electric enthalpy for molecules
and periodic materials and obtains polarization, Born effective charges, and
polarizability by differentiating that same scalar.

This repository provides tools for training MACE-Field from scratch, fine-tuning
MACE foundation models, ASE inference, and batch evaluation.

## What MACE-Field provides

For the energy functional

$$
E(\mathbf{R}, \mathbf{\mathcal{E}}),
$$

the model can return:

**Polarization**

$$
\mathbf{P} = -\frac{1}{\Omega}
\frac{\partial E}{\partial \mathbf{\mathcal{E}}}
$$

**Born effective charges**

$$
Z^*_{\kappa,\alpha\beta} =
\frac{\partial P_\alpha}{\partial R_{\kappa,\beta}}
$$

**Polarizability or susceptibility**

$$
\chi_{\alpha\beta} =
\frac{\partial P_\alpha}{\partial \mathcal{E}_\beta}
$$

The response quantities are derivatives of one scalar model output, so the
energy, forces, and field response remain mutually consistent. At zero field,
the added field coupling contributes zero, allowing the MACE backbone to be
initialized from a foundation model.

## Installation

```bash
git clone https://github.com/mdi-group/mace-field.git
cd mace-field
python -m pip install -e .
```

The generic MACE installation, training, CUDA, and foundation-model guides are
available in the [MACE documentation](https://mace-docs.readthedocs.io/).
For MACE-Field-specific installation, data, training, and inference guidance,
see the [MACE-Field documentation](docs/index.rst).

Build the documentation site locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

## Released MH-0 MACE-Field checkpoint

The [1.0.2 release](https://github.com/mdi-group/mace-field/releases/tag/1.0.2)
includes the [MACE-Field MH-0 checkpoint](https://github.com/mdi-group/mace-field/releases/download/1.0.2/MACEField-MH-0-omat-dielectric.model).
It was initialized from the [MACE multi-head MP-0 foundation model](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1)
and fine-tuned on MP-Dielectric BEC and polarizability labels and MP-Ferroelectric
Berry-phase polarization branches.

The release also includes an [example fine-tuning script](https://github.com/mdi-group/mace-field/releases/download/1.0.2/macefield-example-finetuning-script.sh)
and a [dielectric/ferroelectric replay set](https://github.com/mdi-group/mace-field/releases/download/1.0.2/MP-dielectric-ferroelectric-replay-dataset.xyz).

## Architecture

![MACE-Field architecture](macefield_architecture.png)

MACE-Field extends `ScaleShiftMACE` in `mace.modules.extensions`. The uniform
electric field enters as the equivariant `1o` irrep between interaction layers;
the energy remains the scalar readout.

## Data format

Training data are ASE-readable configurations, normally extended XYZ files.
The default keys are:

### Configuration-level values (`atoms.info`)

| Key | Shape | Units |
| --- | --- | --- |
| `REF_energy` | scalar | eV |
| `REF_stress` | `(6,)` or `(3, 3)` | eV/Å³ |
| `REF_virials` | `(6,)` or `(3, 3)` | eV |
| `REF_electric_field` | `(3,)` | V/Å |
| `REF_polarization` | `(3,)` | e/Å² |
| `REF_polarizability` | `(3, 3)` or `(9,)` | e/(V·Å) |
| `head` | string | multi-head name |

### Per-atom arrays (`atoms.arrays`)

| Key | Shape | Units |
| --- | --- | --- |
| `REF_forces` | `(N, 3)` | eV/Å |
| `REF_becs` | `(N, 3, 3)` or `(N, 9)` | e |

Field targets are required only for the corresponding nonzero loss terms.
Missing field values are represented internally by zeros, but they should not
be used with a nonzero target weight. Key names can be changed with the
`*_key` training options.

## Training from scratch

Use the standard MACE training CLI with the `MACEField` model and
`universal_field` loss:

```bash
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
```

The equivalent source checkout command is:

```bash
python -m mace.cli.run_train [the same options]
```

For periodic systems, the default `UniversalFieldLoss` folds
polarization differences using the cell lattice so that equivalent polarization
branches do not produce artificial discontinuities.

### Replay pseudolabels

`--pseudolabel_replay` generates energy and force labels from the
foundation model, and includes stress/virial labels according to the existing
replay options. Polarization, BEC, and polarizability pseudolabels are enabled
with `--compute_polarization`, `--compute_becs`, and
`--compute_polarizability`; requesting either BECs or polarizability
also requests polarization.

The source checkpoint must itself be MACEField to generate field responses.
MACE-MH-1 is a plain MACE foundation model: it can provide replay energy,
forces, and stress, but it cannot fabricate response labels. A plain source
therefore receives an explicit warning and remains an energy/force/stress-only
replay source. Generated labels are assigned active property weights when the
replayed configuration did not already have them.

## Fine-tuning MACE-MH-1

MACE-MH-1 is a multi-head foundation model. Download the checkpoint from the
[MACE foundation-model release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1)
and select the head appropriate for the target data. For example,
`omat_pbe` is a suitable starting head for OMAT-like inorganic-material
data.

The following fine-tunes one selected MH-1 head into a one-head MACE-Field model:

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

The foundation model's interaction, radial, product, and ordinary energy-readout
parameters are transferred when their shapes are compatible. The MACE-Field
field-coupling parameters are then trained with the selected response targets.
Set `--foundation_model_elements=True` when the target model must
retain all foundation elements.

For replay-based multi-head fine-tuning, set
`--multiheads_finetuning=True`, provide `--pt_train_file`,
and define the new head(s) with `--heads`. The pretrained head is
selected with `--foundation_head`. MACEField keeps the
`universal_field` loss in this mode and enables field outputs for
nonzero field-loss weights.

## Inference

### ASE calculator

```python
from mace.calculators import MACECalculator

calc = MACECalculator(
    model_paths=["MACEField.model"],
    model_type="MACEField",
    electric_field=[0.0, 0.0, 0.02],  # V/Å; overrides atoms.info
    device="cuda",
)

atoms.calc = calc
energy = atoms.get_potential_energy()
polarization = calc.results["polarization"]       # (3,)
becs = calc.results["becs"]                       # (N, 9)
polarizability = calc.results["polarizability"]   # (9,)
```

If `electric_field` is not supplied to the calculator, the field is
read from `atoms.info["electric_field"]`, then
`atoms.info["REF_electric_field"]`. If neither is present, a zero
field is used. For a MACEField calculator, unspecified response-selection
flags retain the historical default of computing polarization, BECs, and
polarizability. To select a subset, pass for example
`compute_polarization=True, compute_becs=False, compute_polarizability=False`; BECs or polarizability always imply
polarization.

### Batch evaluation

```bash
mace_eval_configs \
  --configs=input.xyz \
  --model=MACEField.model \
  --output=output.xyz \
  --compute_polarization \
  --compute_becs \
  --compute_polarizability
```

Use `--electric-field 0 0 0.02` to apply one field to every
configuration, regardless of the per-configuration field stored in the input
file. The output values are written with the `MACE_` prefix by default:

- `atoms.info["MACE_polarization"]`
- `atoms.arrays["MACE_becs"]`
- `atoms.info["MACE_polarizability"]`

### Finite-field ASE workflows

```python
atoms.info["REF_electric_field"] = [0.0, 0.0, 0.1]
atoms.calc = MACECalculator(
    model_paths=["MACEField.model"],
    model_type="MACEField",
    device="cuda",
)

# A time-dependent field can be assigned before each calculation.
atoms.calc.electric_field = [0.0, 0.0, Ez_t]
```

Use this pattern for finite-field relaxation, molecular dynamics,
dielectric-response curves, and ferroelectric switching. The ASE calculator
supports changing the field between steps; the LAMMPS wrappers support a
constant field.

### Fine-tuning selection and active-learning MD

Descriptor-based selection accepts the same field model controls:

```bash
mace_finetuning_select \
  --configs_pt=pretraining.xyz \
  --configs_ft=field_train.xyz \
  --model=MACEField.model \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --output=selected.xyz
```

For committee active-learning MD, use the field controls and opt in to saving
the response tensors:

```bash
mace_active_learning_md \
  --config=initial.xyz \
  --model='MACEField_*.model' \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --save_field_responses \
  --output=trajectory.xyz
```

Saved response names are `MACE_polarization` and
`MACE_polarizability` in `atoms.info`, and
`MACE_becs` in `atoms.arrays`. The active-learning script
uses all three response outputs when saving is enabled and no individual
response flag was selected.

### HDF5 and training visualisation

`mace_prepare_data` preserves field, response, and response-weight tensors in
the repository HDF5 format. The HDF5 writer also accepts `AtomicData` objects
directly, and `HDF5Dataset` reads the resulting files. Select
`--error_table=PerAtomFieldRMSE` to include weighted polarization, BEC, and
polarizability metrics in training tables and scatter plots.

### LAMMPS export

Both supported export formats accept a constant field:

```bash
mace_create_lammps_model MACEField.model \
  --format=libtorch --electric-field 0 0 0.02
mace_create_lammps_model MACEField.model \
  --format=mliap --electric-field 0 0 0.02
```

The LAMMPS wrappers inject a zero field by default and store the configured
field in the exported wrapper. Dynamic per-step field transport is not
supported by this interface; use the ASE calculator when the field changes
between steps.

## Development and tests

Install development dependencies and enable the repository hooks:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

The test suite is organized by capability:

| Directory | Contents |
| --- | --- |
| `tests/unit` | Fast CPU-only unit tests |
| `tests/workflows` | End-to-end CLI training workflows |
| `tests/extensions/macefield` | MACE-Field extension and foundation fine-tuning |
| `tests/extensions/<name>` | Other optional MACE extensions |
| `tests/foundations` | Network-enabled foundation-model tests |
| `tests/integrations` | External-runtime integration tests |

Run the MACE-Field suite locally with:

```bash
python -m pytest tests/extensions/macefield
```

Run the core CPU suites with:

```bash
python -m pytest tests/unit -m "not slow"
python -m pytest tests/workflows
```

The [extension workflow](https://github.com/mdi-group/mace-field/actions/workflows/ci-extensions.yaml)
runs the MACE-Field suite on CPU with small synthetic models. It does not
require network access or a downloaded foundation checkpoint. The
[core](https://github.com/mdi-group/mace-field/actions/workflows/ci-core.yaml)
and [integration](https://github.com/mdi-group/mace-field/actions/workflows/ci-integrations.yaml)
workflows run their respective suites separately.

## References

If you use MACE-Field, please cite:

```bibtex
@article{martin2026macefield,
  title = {General Learning of the Electric Response of Inorganic Materials},
  author = {Martin, Bradley A. A. and Ganose, Alex M. and Kapil, Venkat and
           Li, Tingwei and Butler, Keith T.},
  journal = {PRX Intelligence},
  volume = {1},
  issue = {1},
  pages = {013006},
  numpages = {20},
  year = {2026},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/b116-xy8k},
  url = {https://link.aps.org/doi/10.1103/b116-xy8k}
}
```

and the main MACE papers:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks
         for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kovacs, David Peter and Simm, Gregor N. C. and
          Ortner, Christoph and Csanyi, Gabor},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@misc{Batatia2022Design,
  title={The Design Space of E(3)-Equivariant Atom-Centered Interatomic
         Potentials},
  author={Batatia, Ilyes and Batzner, Simon and Kovacs, David Peter and
          Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and
          Ortner, Christoph and Kozinsky, Boris and Csanyi, Gabor},
  year={2022},
  eprint={2205.06643},
  archivePrefix={arXiv}
}
```

## Acknowledgments

This work has been supported by UKRI funding (EP/Y000552/1 and EP/Y014405/1).

## Contact

- MACE-Field: bradley.martin@ucl.ac.uk
- MACE core: ilyes.batatia@ens-paris-saclay.fr
- Issues and feature requests:
  https://github.com/mdi-group/mace-field/issues
