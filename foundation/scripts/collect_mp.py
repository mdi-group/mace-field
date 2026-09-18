"""Collect Materials Project dielectric/phonon response records.

The MP API does not expose a force/stress record on every dielectric or phonon
document.  This collector therefore writes only properties present on the
same source document and records dielectric tensors as provenance metadata.
It never fabricates missing forces, stresses, or polarizabilities.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DATA_DIR,
    MANIFEST_DIR,
    ensure_workspace,
    manifest_for,
    model_dump,
    set_field_labels,
    set_ref_energy_forces_stress,
    structure_to_atoms,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_mp")
BASE_URL = "https://api.materialsproject.org"


def _key() -> str:
    key = os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
    if not key:
        raise SystemExit(
            "Set MP_API_KEY (the key is read only from the environment and is "
            "never written to the workspace)."
        )
    return key


def _field(document: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in document:
            return document[name]
    return None


def _search(
    rest,
    fields: list[str],
    chunks: int | None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    values = rest.search(
        num_chunks=chunks,
        chunk_size=1000,
        all_fields=False,
        fields=fields,
        **kwargs,
    )
    return [model_dump(value) for value in values]


def collect(output: Path, *, chunks: int | None, dielectric_as_polarizability: bool) -> dict[str, Any]:
    ensure_workspace()
    from mp_api.client import MPRester

    # MPRester reads MP_API_KEY itself, but passing it makes the failure mode
    # deterministic and avoids a warning that can expose a secret in logs.
    with MPRester(api_key=_key()) as mpr:
        dielectric_fields = [
            "material_id",
            "total",
            "ionic",
            "electronic",
            "e_total",
            "e_ionic",
            "e_electronic",
        ]
        dielectric = _search(mpr.materials.dielectric, dielectric_fields, chunks)
        dielectric_ids = [
            str(_field(record, "material_id", "identifier"))
            for record in dielectric
            if _field(record, "material_id", "identifier") is not None
        ]
        summaries: list[dict[str, Any]] = []
        for start in range(0, len(dielectric_ids), 500):
            try:
                summaries.extend(
                    _search(
                        mpr.materials.summary,
                        ["material_id", "structure", "energy_per_atom"],
                        chunks=None,
                        material_ids=dielectric_ids[start : start + 500],
                    )
                )
            except Exception as error:  # A summary route outage should not lose responses.
                LOG.warning("MP summary structure lookup failed for batch %d: %s", start // 500, error)
        try:
            phonon_fields = [
                "identifier",
                "structure",
                "born",
                "epsilon_static",
                "epsilon_electronic",
                "total_dft_energy",
            ]
            phonon = _search(mpr.materials.phonon, phonon_fields, chunks)
        except Exception as error:  # API route support differs by deployment.
            LOG.warning("MP phonon route was unavailable: %s", error)
            phonon = []

    by_id = {str(_field(record, "material_id", "identifier")): record for record in dielectric}
    for record in summaries:
        identifier = _field(record, "material_id", "identifier")
        if identifier is not None:
            by_id.setdefault(str(identifier), {}).update(record)
    for record in phonon:
        identifier = _field(record, "material_id", "identifier")
        if identifier is not None:
            by_id.setdefault(str(identifier), {}).update(record)

    frames = []
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    for source_id, record in sorted(by_id.items()):
        structure = _field(record, "structure")
        if structure is None:
            continue
        try:
            atoms = structure_to_atoms(structure)
        except (TypeError, ValueError, KeyError) as error:
            LOG.warning("Skipping %s: structure conversion failed: %s", source_id, error)
            continue
        atoms.info["source"] = "Materials Project dielectric/phonon API"
        atoms.info["source_material_id"] = source_id

        before = set(atoms.info) | set(atoms.arrays)
        set_ref_energy_forces_stress(
            atoms,
            energy=_field(record, "total_dft_energy"),
            forces=_field(record, "forces"),
            stress=_field(record, "stress"),
        )
        static = _field(record, "epsilon_static", "total")
        electronic = _field(record, "epsilon_electronic", "electronic")
        if static is not None:
            atoms.info["MP_epsilon_static"] = np.asarray(static, dtype=float).tolist()
        if electronic is not None:
            atoms.info["MP_epsilon_electronic"] = np.asarray(electronic, dtype=float).tolist()

        # Born charges are only directly usable when the API provides one for
        # every atom.  Phonon documents commonly contain symmetry-unique atoms.
        born = _field(record, "born")
        becs = None if born is None else np.asarray(born, dtype=float)
        if becs is not None and becs.shape == (len(atoms), 3, 3):
            labels = set_field_labels(atoms, electric_field=[0.0, 0.0, 0.0], becs=becs)
            counts["becs"] += int(labels["becs"])
        if dielectric_as_polarizability and static is not None:
            epsilon = np.asarray(static, dtype=float)
            if epsilon.shape == (3, 3):
                # MACE-Field's response head uses the susceptibility relative
                # to eps0.  This opt-in conversion is intentionally not the
                # default because MP dielectric tensors and field-label units
                # must be checked against the selected training convention.
                labels = set_field_labels(
                    atoms,
                    electric_field=[0.0, 0.0, 0.0],
                    polarizability=epsilon - np.eye(3),
                )
                counts["polarizability"] += int(labels["polarizability"])

        after = set(atoms.info) | set(atoms.arrays)
        for label, key in (("energy", "REF_energy"), ("forces", "REF_forces"), ("stress", "REF_stress")):
            counts[label] += int(key in after and key not in before)
        frames.append(atoms)

    if not frames:
        raise RuntimeError("MP returned no structures; check API access and route fields")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="MP-Dielectric",
        output=output,
        frames=written,
        labels=counts,
        sources=[
            {"url": BASE_URL + "/materials/dielectric", "records": len(dielectric)},
            {"url": BASE_URL + "/materials/phonon", "records": len(phonon)},
        ],
        notes=[
            "Energy, forces, and stress are retained only when present on the same API document.",
            "Symmetry-reduced Born charges are not expanded; only complete atom-wise arrays are labelled.",
            "MP dielectric tensors are metadata by default; use --dielectric-as-polarizability only after validating the training-unit convention.",
            "The raw union may include phonon records without BEC or dielectric labels; the cleaning stage removes response-free MP-Dielectric frames from training.",
        ],
    )
    write_json(MANIFEST_DIR / "MP-Dielectric.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "MP-Dielectric.xyz")
    parser.add_argument("--chunks", type=int, default=None, help="Maximum 1000-record API chunks per route")
    parser.add_argument("--dielectric-as-polarizability", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(args.output, chunks=args.chunks, dielectric_as_polarizability=args.dielectric_as_polarizability)
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
