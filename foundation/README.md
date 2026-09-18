# MACE-MH-1 → MACEField foundation workspace

This untracked directory prepares field-response data and a reproducible
multi-head fine-tuning launch for the repository’s MACE-MH-1 foundation model.
It is separate from the existing untracked root `data/`, `inference/`, and
`notebooks/` directories. Generated datasets, source caches, checkpoints, and
logs are intentionally not part of the repository hygiene commits.

## Output contract

The field-aware training path uses the following canonical keys:

| Location | Key | Shape | Units/convention |
| --- | --- | --- | --- |
| `Atoms.info` | `REF_energy` | scalar | eV |
| `Atoms.arrays` | `REF_forces` | `(N, 3)` | eV/Å |
| `Atoms.info` | `REF_stress` | `(6,)`, `(9,)`, or `(3, 3)` | eV/Å³, ASE sign convention |
| `Atoms.info` | `REF_electric_field` | `(3,)` | V/Å |
| `Atoms.info` | `REF_polarization` | `(3,)` | e/Å² |
| `Atoms.arrays` | `REF_becs` | `(N, 9)` or `(N, 3, 3)` | e |
| `Atoms.info` | `REF_polarizability` | `(9,)` or `(3, 3)` | dimensionless susceptibility, `εr - I` |

The model returns polarization, BECs, and polarizability by differentiating the
same scalar field-dependent energy. The polarizability output is normalized by
cell volume and vacuum permittivity in `MACEField`; therefore dielectric tensors
are converted to `εr - I` before being used as this target. A 3-D
per-configuration convention is not inferred from C2DB’s 2-D polarizability or
spontaneous-polarization values; those remain provenance metadata.

The extxyz boundary cannot safely store a three-dimensional per-atom array, so
`REF_becs` is written as `(N, 9)`. MACE accepts both forms and the in-memory
field shape is `(N, 3, 3)`.

## Collected sources

Each successful collector writes a manifest under `foundation/manifests/` with
frame counts, label counts, source URLs, and a SHA-256 checksum.

- `MP-Dielectric.xyz` combines Materials Project dielectric and phonon API
  records. Complete atom-wise BECs and `εr - I` polarizability are retained;
  E/F/stress are retained only when present on that same API document. The MP
  task API exposes metadata and parsed output, but does not expose a public
  OUTCAR file tree. `audit_mp_tasks.py` records the fields returned for task
  pages such as `aaadjozp` and the current `MPRester.get_download_info` result.
- `MP-ferroelectric.xyz` reads the MPContribs `ferroelectrics` project. Its
  `workflow.json.gz` attachments contain the matching pymatgen structures,
  energies, forces, stresses, and same-branch Berry polarization. These exact
  per-workflow records are preferred; structure-only polarization fallbacks
  are used only when no workflow attachment is available. Polarization is
  converted from µC/cm² to e/Å², and attachment VASP stresses from kbar to
  ASE/MACE eV/Å³ with the VASP sign convention.
- `finite-field-ferroelectric.extxyz` is copied unchanged from the repository’s
  `data/` directory. Its source-specific
  `REF_total_polarisation` key is mapped explicitly in the fine-tuning head;
  MACEField permits this field-only head without fabricating E/F/stress labels.
- `JarvisDB.xyz` is parsed from all available archives in the public
  [JARVIS DFPT raw Figshare project](https://figshare.com/projects/JARVIS-DFT_DFPT_raw_input_output_files/82118).
  Complete OUTCAR BEC blocks are required; same-archive energy, forces, and
  stress are retained when ASE parses them. Static dielectric tensors are
  stored as `εr - I`. Ordinary JARVIS `dft_3d` summary records are not included
  in this field-response dataset.
- `C2DB.xyz` collects all live-site candidates with dielectric or spontaneous
  polarization/BEC data. Its public XYZ download supplies E/F/stress and its
  result files supply complete BECs. The source’s 2-D alpha and polarization
  values remain explicitly named metadata.
- `Togo.xyz` collects the NIMS MDR Togo phonon collection. The maintained
  [PhononDB index](https://github.com/atztogo/phonondb/blob/main/mdr/phonondb/README.md)
  is used by default because it lists all 10,034 direct archive links; the
  slower MDR HTML pagination remains available with `--html-index`. Complete
  BEC arrays from `phonopy_params.yaml.xz` and compatible relative dielectric
  tensors are retained. Harmonic phonon archives do not generally contain
  matching E/F/stress labels.

Primary source pages: [Materials Project API](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started),
[MPContribs](https://docs.materialsproject.org/services/mpcontribs),
[JARVIS dft_3d](https://jarvis-materials-design.github.io/dbdocs/jarvisdft/),
[C2DB](https://c2db.fysik.dtu.dk/), and the
[NIMS MDR API manual](https://dice.nims.go.jp/services/MDR/manual/html/api.html).

## Collection and validation

Run from the repository root. The MP collectors read `MP_API_KEY` only from
the environment; the key is never written to a file or log.

```bash
python foundation/scripts/collect_all.py --continue-on-error
python foundation/scripts/audit_mp_tasks.py
python foundation/scripts/clean_validate_datasets.py
python foundation/scripts/validate_datasets.py foundation/data/cleaned/*.xyz foundation/data/cleaned/*.extxyz \
  --output foundation/manifests/cleaned_shape_summary.json
python foundation/scripts/make_replay_set.py
```

`collect_all.py` uses the raw JARVIS DFPT collector by default. Use
`--jarvis-summary` when only the lightweight summary archive is wanted. Use
`--max-c2db`, `--max-togo`, and `--max-jarvis` for bounded schema smoke tests;
omit them for a complete public-source attempt. `--jarvis-workers`,
`--togo-workers`, and `--c2db-workers` control bounded parallelism. The Togo
collector uses the maintained PhononDB GitHub index by default; use
`--togo-html-index` only when you specifically need live MDR pagination.
Source downloads are resumable in `foundation/.cache/`.

The individual collectors are:

```bash
MP_API_KEY="$MP_API_KEY" python foundation/scripts/collect_mp.py --dielectric-as-polarizability
MP_API_KEY="$MP_API_KEY" python foundation/scripts/collect_mp_contribs.py
python foundation/scripts/copy_finite_field.py
python foundation/scripts/collect_jarvis_dfpt.py
python foundation/scripts/collect_c2db.py
python foundation/scripts/collect_togo.py
```

`clean_validate_datasets.py` writes audited copies under
`foundation/data/cleaned/` and leaves the raw source files unchanged. It
rejects non-finite or malformed labels, extreme values, and BECs whose maximum
absolute acoustic sum-rule residual exceeds `0.25 e` by default. Its manifest
also records units, per-label counts, same-frame E/F/stress-to-response
co-occurrence, and correlations of per-frame label norms; these correlations
are diagnostics, not cross-structure label matching. The training config and
replay builder prefer these cleaned copies automatically.

Response-labeled structures over 128 atoms are excluded from the cleaned
training copies. MACE-MH-1-sized response heads require differentiable second
derivatives for BECs and polarizabilities, and this bound keeps the default
four-GPU run within 24 GB per GPU at batch size two. Raw source files and their
manifests remain available for a less conservative downstream policy.

`make_replay_set.py` removes source target fields and creates the deterministic
`foundation/data/mh1_replay.xyz` and `mh1_replay_valid.xyz`. The replay head is
structure-only at input time: MACE-MH-1 supplies E/F/(optional stress)
pseudolabels. A plain MACE-MH-1 source cannot generate field responses, so real
polarization/BEC/polarizability labels must come from the field-capable heads.

## Four-GPU fine-tuning

After collection and replay generation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MACE_MAX_EPOCHS=2048 \
bash foundation/run_mh1_macefield_4gpu.sh
```

The launcher selects `omat_pbe` from the `mh-1` foundation, constructs a
`MACEField` model with `universal_field`, enables all response outputs, and
uses MACE-MH-1 only for plain replay pseudolabels. Additional
`mace.cli.run_train` options can be appended to the script. For a memory
smoke test, run one epoch first with `MACE_MAX_EPOCHS=1`; the launcher defaults
to batch size two with an atom budget and exposes `MACE_BATCH_SIZE`,
`MACE_VALID_BATCH_SIZE`, and `MACE_MAX_ATOMS_PER_BATCH` for further tuning.
Response computation is gated per batch by active label
weights, so replay and unrelated E/F/P batches do not build unused BEC or
polarizability graphs. `PYTORCH_ALLOC_CONF` can be supplied to override the
allocator default.
The launcher defaults to training batch size two (validation batch size one)
with a 288-atom budget.  The atom budget prevents mixed-size response batches
from exceeding GPU memory; override it with `MACE_MAX_ATOMS_PER_BATCH` when
running on different hardware.  The configured real heads are:

| Head | Main labels |
| --- | --- |
| `MP-ferroelectric` | Berry polarization plus exact attachment E/F/stress |
| `finite-field-ferroelectric` | source total polarization and electric field |
| `C2DB` | E/F/stress and complete BECs where available |
| `MP-Dielectric` | complete BECs and `εr - I` where available |
| `JarvisDB` | raw DFPT BECs, `εr - I`, and same-archive E/F/stress where available |
| `Togo` | complete DFPT BECs and compatible `εr - I` where available |
| `pt_head` | MACE-MH-1 E/F/stress pseudolabel replay |

Missing properties receive zero per-configuration weights; they are not
fabricated zeros. Generic MACE installation, foundation-model downloads,
training options, and device setup are documented upstream in [MACE](https://github.com/ACEsuit/mace).
