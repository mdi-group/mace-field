"""Collect Berry-polarization structures from an MPContribs project."""

from __future__ import annotations

import argparse
import gzip
import json
import base64
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import units as ase_units

from common import (
    CACHE_DIR,
    DATA_DIR,
    MANIFEST_DIR,
    MICROCOULOMB_PER_CM2_TO_E_PER_A2,
    ensure_workspace,
    manifest_for,
    set_field_labels,
    set_ref_energy_forces_stress,
    structure_to_atoms,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_mp_contribs")
PROJECT_URL = "https://contribs.materialsproject.org/projects/ferroelectrics"
# The workflow attachment stores the VASP stress tensor in kbar.  ASE's VASP
# reader converts it to its eV/A^3 convention with a sign reversal.
VASP_KBAR_TO_ASE_STRESS = -0.1 * ase_units.GPa


def _key() -> str:
    key = os.environ.get("MP_API_KEY") or os.environ.get("MPCONTRIBS_API_KEY")
    if not key:
        raise SystemExit(
            "Set MP_API_KEY or MPCONTRIBS_API_KEY (the key is read only from "
            "the environment and is never written to the workspace)."
        )
    return key


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, path + (str(index),))


def _polarization_vector(data: dict[str, Any]) -> np.ndarray | None:
    for path, value in _walk(data):
        if "polar" not in " ".join(path).lower():
            continue
        candidate = value
        if isinstance(value, dict):
            for key in ("vector", "value", "cartesian"):
                if key in value:
                    candidate = value[key]
                    break
            else:
                continue
        if isinstance(candidate, dict):
            candidate = [
                candidate.get(axis, {}).get("value")
                if isinstance(candidate.get(axis), dict)
                else candidate.get(axis)
                for axis in ("a", "b", "c")
            ]
        try:
            array = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size == 3 and np.all(np.isfinite(array)):
            return array.reshape(3) * MICROCOULOMB_PER_CM2_TO_E_PER_A2
    return None


def _structure_ids(contribution: dict[str, Any]) -> list[str]:
    result = []
    for value in contribution.get("structures", []) or []:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, dict):
            identifier = value.get("id") or value.get("_id") or value.get("structure_id")
            if identifier is not None:
                result.append(str(identifier))
    return result


def _load_structure_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    structures = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("structures", [payload])
        for structure in payload:
            identifier = structure.get("id") or structure.get("_id")
            if identifier is not None:
                structures[str(identifier)] = structure
    return structures


def _load_attachment_payloads(paths: Iterable[Path]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Decode MPContribs' outer attachment envelope and inner gzip JSON."""

    payloads = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            outer = json.load(handle)
        for item in outer:
            content = item.get("content")
            if not content:
                continue
            try:
                inner = json.loads(gzip.decompress(base64.b64decode(content)))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(inner, dict):
                payloads[str(item.get("id", ""))] = (str(item.get("name", "")), inner)
    return payloads


def _attachment_frames(
    contributions: list[dict[str, Any]],
    attachment_payloads: dict[str, tuple[str, dict[str, Any]]],
) -> tuple[list[Any], set[str], dict[str, int]]:
    """Build exact E/F/stress/polarization frames from workflow attachments."""

    frames = []
    covered_contributions: set[str] = set()
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0}
    for contribution in contributions:
        contribution_id = str(contribution.get("id", contribution.get("_id", "")))
        for attachment in contribution.get("attachments", []) or []:
            attachment_id = str(attachment.get("id", ""))
            name, workflow = attachment_payloads.get(attachment_id, ("", {}))
            if "workflow" not in name.lower():
                continue
            structures = workflow.get("structures") or []
            polarizations = workflow.get("same_branch_polarization") or []
            energies = workflow.get("energies") or []
            forces = workflow.get("forces") or []
            stresses = workflow.get("stresses") or []
            nframes = min(len(structures), len(polarizations))
            if nframes == 0:
                continue
            covered_contributions.add(contribution_id)
            for index in range(nframes):
                try:
                    atoms = structure_to_atoms(structures[index])
                    vector = np.asarray(polarizations[index], dtype=float).reshape(3)
                except (TypeError, ValueError, KeyError):
                    continue
                atoms.info["source"] = "MPContribs ferroelectrics workflow attachment"
                atoms.info["source_contribution_id"] = contribution_id
                atoms.info["source_workflow_index"] = int(index)
                set_ref_energy_forces_stress(
                    atoms,
                    energy=energies[index] if index < len(energies) else None,
                    forces=forces[index] if index < len(forces) else None,
                    stress=(
                        np.asarray(stresses[index], dtype=float) * VASP_KBAR_TO_ASE_STRESS
                        if index < len(stresses)
                        else None
                    ),
                )
                labels = set_field_labels(
                    atoms,
                    electric_field=[0.0, 0.0, 0.0],
                    polarization=vector * MICROCOULOMB_PER_CM2_TO_E_PER_A2,
                )
                counts["polarization"] += int(labels["polarization"])
                counts["energy"] += int("REF_energy" in atoms.info)
                counts["forces"] += int("REF_forces" in atoms.arrays)
                counts["stress"] += int("REF_stress" in atoms.info)
                frames.append(atoms)
    return frames, covered_contributions, counts


def collect(output: Path, *, project: str, chunk_size: int) -> dict[str, Any]:
    ensure_workspace()
    from mpcontribs.client import Client

    client = Client(apikey=_key(), project=project)
    contribution_result = client.query_contributions(fields=["_all"], paginate=True)
    if isinstance(contribution_result, dict):
        contributions = list(contribution_result.get("data", []))
    else:
        contributions = list(contribution_result)
    # The workflow attachments contain the structures, E/F/stress trajectory,
    # and same-branch Berry polarization on one exact calculation path.  They
    # are preferred over the lighter structure endpoint whenever available.
    attachment_ids = []
    for contribution in contributions:
        for attachment in contribution.get("attachments", []) or []:
            identifier = str(attachment.get("id", ""))
            if identifier and identifier not in attachment_ids:
                attachment_ids.append(identifier)
    attachment_cache = CACHE_DIR / "mpcontribs" / project / "attachments"
    attachment_paths = []
    for start in range(0, len(attachment_ids), chunk_size):
        attachment_paths.extend(
            client.download_attachments(
                attachment_ids[start : start + chunk_size],
                outdir=attachment_cache,
                overwrite=False,
                fmt="json",
            )
        )
    attachment_payloads = _load_attachment_payloads(attachment_paths)
    frames, covered_contributions, counts = _attachment_frames(contributions, attachment_payloads)

    ids = []
    for contribution in contributions:
        for identifier in _structure_ids(contribution):
            if identifier not in ids:
                ids.append(identifier)
    cache = CACHE_DIR / "mpcontribs" / project
    paths = []
    for start in range(0, len(ids), chunk_size):
        batch = ids[start : start + chunk_size]
        paths.extend(client.download_structures(batch, outdir=cache, overwrite=False, fmt="json"))
    structure_map = _load_structure_files(paths)

    # Index each structure by the first contribution carrying a polarization
    # vector.  If a project publishes several branches, prefer a structure
    # whose name explicitly identifies the polar branch.
    selected: dict[str, tuple[np.ndarray, str]] = {}
    for contribution in contributions:
        vector = _polarization_vector(contribution.get("data", contribution))
        if vector is None:
            continue
        identifiers = _structure_ids(contribution)
        polar_ids = []
        for identifier in identifiers:
            name = str(structure_map.get(identifier, {}).get("name", "")).lower()
            if "polar" in name and "nonpolar" not in name:
                polar_ids.append(identifier)
        for identifier in polar_ids or identifiers[:1]:
            selected.setdefault(identifier, (vector, str(contribution.get("id", contribution.get("_id", "")))))

    fallback_counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    for identifier, (vector, contribution_id) in selected.items():
        if contribution_id in covered_contributions:
            continue
        structure = structure_map.get(identifier)
        if structure is None:
            LOG.warning("Structure %s was not returned by MPContribs", identifier)
            continue
        try:
            atoms = structure_to_atoms(structure)
        except (TypeError, ValueError, KeyError) as error:
            LOG.warning("Skipping %s: structure conversion failed: %s", identifier, error)
            continue
        atoms.info["source"] = "MPContribs ferroelectrics"
        atoms.info["source_structure_id"] = identifier
        atoms.info["source_contribution_id"] = contribution_id
        labels = set_field_labels(
            atoms,
            electric_field=[0.0, 0.0, 0.0],
            polarization=vector,
        )
        fallback_counts["polarization"] += int(labels["polarization"])
        frames.append(atoms)

    for label in fallback_counts:
        counts[label] = counts.get(label, 0) + fallback_counts[label]

    if not frames:
        raise RuntimeError("MPContribs returned no structures with polarization vectors")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="MP-ferroelectric",
        output=output,
        frames=written,
        labels=counts,
        sources=[{"url": PROJECT_URL, "project": project, "contributions": len(contributions)}],
        notes=[
            "Polarization vectors are converted from microC/cm^2 to e/Angstrom^2.",
            "Workflow attachment stresses are converted from VASP kbar to ASE/MACE eV/Angstrom^3 with the VASP sign convention.",
            "MPContribs does not provide an exact matching force/stress record for every structure; those labels are intentionally absent.",
            "The project vector is retained in the source frame; verify Cartesian-vs-lattice convention before quantitative training.",
        ],
    )
    write_json(MANIFEST_DIR / "MP-ferroelectric.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ferroelectrics")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "MP-ferroelectric.xyz")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(args.output, project=args.project, chunk_size=args.chunk_size)
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
