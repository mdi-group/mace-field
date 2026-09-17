<p align="center">
  <img src="macefield_logo.png" alt="MACE-Field logo" width="600">
</p>

# MACE-Field: Electric-Field-Aware MACE Models

MACE-Field extends the [MACE](https://github.com/ACEsuit/mace) architecture with a
uniform electric-field input. It learns a scalar electric enthalpy for molecules
and periodic materials and obtains polarization, Born effective charges, and
polarizability by differentiating that same scalar.

This repository contains the MACE-Field extension on top of the current MACE
architecture. It supports training from scratch, fine-tuning MACE foundation
models, ASE inference, and batch evaluation.

## What MACE-Field provides

For an energy functional \(E(\mathbf{R}, \mathbf{\mathcal{E}})\), the model can
return:

- Polarization:
  \( \mathbf{P} = -\frac{1}{\Omega}
  \frac{\partial E}{\partial \mathbf{\mathcal{E}}} \)
- Born effective charges:
  \( Z^*_{\kappa,\alpha\beta} =
  \frac{\partial P_\alpha}{\partial R_{\kappa,\beta}} \)
- Polarizability/susceptibility:
  \( \chi_{\alpha\beta} =
  \frac{\partial P_\alpha}{\partial \mathcal{E}_\beta} \)

The response quantities are derivatives of one scalar model output, so the
energy, forces, and field response remain mutually consistent. At zero field,
the added field coupling contributes zero, allowing the MACE backbone to be
initialized from a foundation model.

## Installation

~~~bash
git clone https://github.com/mdi-group/mace-field.git
cd mace-field
python -m pip install -e .
~~~

The generic MACE installation, training, CUDA, and foundation-model guides are
available in the [MACE documentation](https://mace-docs.readthedocs.io/).

## Architecture

![MACE-Field architecture](macefield_architecture.png)

MACE-Field is implemented as an extension of <code>ScaleShiftMACE</code> in
<code>mace.modules.extensions</code>. The uniform electric field is represented
as the equivariant <code>1o</code> irrep and is coupled to the latent features
between interaction layers. The standard MACE energy readout remains the scalar
output.

## Data format

Training data are ASE-readable configurations, normally extended XYZ files.
The default keys are:

### Configuration-level values (<code>atoms.info</code>)

| Key | Shape | Units |
| --- | --- | --- |
| <code>REF_energy</code> | scalar | eV |
| <code>REF_stress</code> | <code>(6,)</code> or <code>(3, 3)</code> | eV/Å³ |
| <code>REF_virials</code> | <code>(6,)</code> or <code>(3, 3)</code> | eV |
| <code>REF_electric_field</code> | <code>(3,)</code> | V/Å |
| <code>REF_polarization</code> | <code>(3,)</code> | e/Å² |
| <code>REF_polarizability</code> | <code>(3, 3)</code> or <code>(9,)</code> | e/(V·Å) |
| <code>head</code> | string | multi-head name |

### Per-atom arrays (<code>atoms.arrays</code>)

| Key | Shape | Units |
| --- | --- | --- |
| <code>REF_forces</code> | <code>(N, 3)</code> | eV/Å |
| <code>REF_becs</code> | <code>(N, 3, 3)</code> or <code>(N, 9)</code> | e |

Field targets are required only for the corresponding nonzero loss terms.
Missing field values are represented internally by zeros, but they should not
be used with a nonzero target weight. Key names can be changed with the
<code>*_key</code> training options.

## Training from scratch

Use the standard MACE training CLI with the <code>MACEField</code> model and
<code>universal_field</code> loss:

~~~bash
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
~~~

The equivalent source checkout command is:

~~~bash
python -m mace.cli.run_train [the same options]
~~~

For periodic systems, the default <code>UniversalFieldLoss</code> folds
polarization differences using the cell lattice so that equivalent polarization
branches do not produce artificial discontinuities.

### Replay pseudolabels

<code>--pseudolabel_replay</code> generates energy and force labels from the
foundation model, and includes stress/virial labels according to the existing
replay options. Polarization, BEC, and polarizability pseudolabels are enabled
with <code>--compute_polarization</code>, <code>--compute_becs</code>, and
<code>--compute_polarizability</code>; requesting either BECs or polarizability
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
<code>omat_pbe</code> is a suitable starting head for OMAT-like inorganic-material
data.

The following fine-tunes one selected MH-1 head into a one-head MACE-Field model:

~~~bash
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
~~~

The foundation model's interaction, radial, product, and ordinary energy-readout
parameters are transferred when their shapes are compatible. The MACE-Field
field-coupling parameters are then trained with the selected response targets.
Set <code>--foundation_model_elements=True</code> when the target model must
retain all foundation elements.

For replay-based multi-head fine-tuning, set
<code>--multiheads_finetuning=True</code>, provide <code>--pt_train_file</code>,
and define the new head(s) with <code>--heads</code>. The pretrained head is
selected with <code>--foundation_head</code>. MACEField keeps the
<code>universal_field</code> loss in this mode and enables field outputs for
nonzero field-loss weights.

## Inference

### ASE calculator

~~~python
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
~~~

If <code>electric_field</code> is not supplied to the calculator, the field is
read from <code>atoms.info["electric_field"]</code>, then
<code>atoms.info["REF_electric_field"]</code>. If neither is present, a zero
field is used. For a MACEField calculator, unspecified response-selection
flags retain the historical default of computing polarization, BECs, and
polarizability. To select a subset, pass for example
<code>compute_polarization=True, compute_becs=False,
compute_polarizability=False</code>; BECs or polarizability always imply
polarization.

### Batch evaluation

~~~bash
mace_eval_configs \
  --configs=input.xyz \
  --model=MACEField.model \
  --output=output.xyz \
  --compute_polarization \
  --compute_becs \
  --compute_polarizability
~~~

Use <code>--electric-field Ex Ey Ez</code> to apply one field to every
configuration, regardless of the per-configuration field stored in the input
file. The output values are written with the <code>MACE_</code> prefix by default:

- <code>atoms.info["MACE_polarization"]</code>
- <code>atoms.arrays["MACE_becs"]</code>
- <code>atoms.info["MACE_polarizability"]</code>

### Finite-field ASE workflows

~~~python
atoms.info["REF_electric_field"] = [0.0, 0.0, 0.1]
atoms.calc = MACECalculator(
    model_paths=["MACEField.model"],
    model_type="MACEField",
    device="cuda",
)

# A time-dependent field can be assigned before each calculation.
atoms.calc.electric_field = [0.0, 0.0, Ez_t]
~~~

This supports finite-field relaxation, molecular dynamics, dielectric-response
curves, and ferroelectric switching workflows. The current documented
field-aware path is the ASE calculator; the former environment-variable example
is not consumed by the current MLIAP wrapper.

### Fine-tuning selection and active-learning MD

Descriptor-based selection accepts the same field model controls:

~~~bash
mace_finetuning_select \
  --configs_pt=pretraining.xyz \
  --configs_ft=field_train.xyz \
  --model=MACEField.model \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --output=selected.xyz
~~~

For committee active-learning MD, use the field controls and opt in to saving
the response tensors:

~~~bash
mace_active_learning_md \
  --config=initial.xyz \
  --model='MACEField_*.model' \
  --model_type=MACEField \
  --electric-field 0 0 0.02 \
  --save_field_responses \
  --output=trajectory.xyz
~~~

Saved response names are <code>MACE_polarization</code> and
<code>MACE_polarizability</code> in <code>atoms.info</code>, and
<code>MACE_becs</code> in <code>atoms.arrays</code>. The active-learning script
uses all three response outputs when saving is enabled and no individual
response flag was selected.

### HDF5 and training visualisation

<code>mace_prepare_data</code> preserves the field, response, and response-weight
tensors in the repository HDF5 format. The public HDF5 writer also accepts
<code>AtomicData</code> objects directly, and its output can be read back by
<code>HDF5Dataset</code>. Select <code>--error_table=PerAtomFieldRMSE</code> to
include weighted polarization, BEC, and polarizability metrics in the training
table and scatter plots.

### LAMMPS export

Both supported export formats accept a constant field:

~~~bash
mace_create_lammps_model MACEField.model \
  --format=libtorch --electric-field 0 0 0.02
mace_create_lammps_model MACEField.model \
  --format=mliap --electric-field 0 0 0.02
~~~

The LAMMPS wrappers inject a zero field by default and store the configured
field in the exported wrapper. Dynamic per-step field transport is not
supported by this interface; use the ASE calculator when the field changes
between steps.

<code>mace_polar_density_cube</code> is specific to PolarMACE electrostatic
density outputs and is not a MACEField response interface.

## Development and tests

Install development dependencies and enable the repository hooks:

~~~bash
python -m pip install -e '.[dev]'
pre-commit install
~~~

The test suite is organized by capability:

| Directory | Contents |
| --- | --- |
| <code>tests/unit</code> | Fast CPU-only unit tests |
| <code>tests/workflows</code> | End-to-end CLI training workflows |
| <code>tests/extensions/macefield</code> | MACE-Field extension and foundation fine-tuning |
| <code>tests/extensions/&lt;name&gt;</code> | Other optional MACE extensions |
| <code>tests/foundations</code> | Network-enabled foundation-model tests |
| <code>tests/integrations</code> | External-runtime integration tests |

Run the MACE-Field suite locally with:

~~~bash
python -m pytest tests/extensions/macefield
~~~

Run the core CPU suites with:

~~~bash
python -m pytest tests/unit -m "not slow"
python -m pytest tests/workflows
~~~

CI runs the MACE-Field suite as a dedicated CPU extension job. The suite uses
small synthetic models and does not require network access or a foundation
checkpoint download.

## References

If you use MACE-Field, please cite:

~~~bibtex
@misc{martin2025generallearningelectricresponse,
  title={General Learning of the Electric Response of Inorganic Materials},
  author={Martin, Bradley A. A. and Ganose, Alex M. and Kapil, Venkat and
           Li, Tingwei and Butler, Keith T.},
  year={2025},
  eprint={2508.17870},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci}
}
~~~

and the main MACE papers:

~~~bibtex
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
~~~

## Acknowledgments

This work has been supported by UKRI funding (EP/Y000552/1 and EP/Y014405/1).

## Contact

- MACE-Field: bradley.martin@ucl.ac.uk
- MACE core: ilyes.batatia@ens-paris-saclay.fr
- Issues and feature requests:
  https://github.com/mdi-group/mace-field/issues
