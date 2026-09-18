"""Collect structures and response metadata from the public JARVIS archives.

The published JARVIS dft_3d summary archive contains structures, energies,
and dielectric scalars, but does not contain atom-wise Born charges or force
and stress arrays.  Those fields are kept as metadata unless a record carries
a complete, shape-valid MACE label directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from common import (
    DATA_DIR,
    MANIFEST_DIR,
    as_numpy,
    download_file,
    ensure_workspace,
    manifest_for,
    set_field_labels,
    set_ref_energy_forces_stress,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_jarvis")
ARCHIVES = {
    "dft_3d": {
        "url": "https://ndownloader.figshare.com/files/38521619",
        "member": "jdft_3d-12-12-2022.json",
    },
    "c2db": {
        "url": "https://ndownloader.figshare.com/files/28682010",
        "member": "c2db_atoms.json",
    },
}


def _atoms(record: dict[str, Any]) -> Atoms | None:
    payload = record.get("atoms", record)
    try:
        elements = payload.get("elements") or payload.get("symbols")
        coords = payload.get("coords") or payload.get("positions")
        lattice = payload.get("lattice_mat") or payload.get("cell")
        if elements is None or coords is None:
            return None
        return Atoms(
            symbols=list(elements),
            positions=np.asarray(coords, dtype=float),
            cell=None if lattice is None else np.asarray(lattice, dtype=float),
            pbc=False if lattice is None else True,
        )
    except (TypeError, ValueError):
        return None


def _find_member(archive: zipfile.ZipFile, requested: str) -> str:
    names = archive.namelist()
    for name in names:
        if name.endswith(requested):
            return name
    json_names = [name for name in names if name.endswith(".json")]
    if len(json_names) == 1:
        return json_names[0]
    raise FileNotFoundError(f"Could not find {requested} in {archive.filename}")


def collect(output: Path, *, dataset: str, max_records: int | None) -> dict[str, Any]:
    ensure_workspace()
    if dataset not in ARCHIVES:
        raise ValueError(f"Unknown JARVIS dataset {dataset}; choose from {sorted(ARCHIVES)}")
    source = ARCHIVES[dataset]
    archive_path = Path(__file__).resolve().parents[1] / ".cache" / "jarvis" / f"{dataset}.zip"
    download_file(source["url"], archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_member(archive, source["member"])
        with archive.open(member) as handle:
            records = json.load(handle)
    if max_records is not None:
        records = records[:max_records]

    frames = []
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    for record in records:
        atoms = _atoms(record)
        if atoms is None:
            continue
        atoms.info["source"] = f"JARVIS {dataset} Figshare archive"
        if record.get("jid") is not None:
            atoms.info["source_jid"] = str(record["jid"])
        for key in ("epsx", "epsy", "epsz", "mepsx", "mepsy", "mepsz"):
            if record.get(key) is not None:
                try:
                    atoms.info[f"JARVIS_{key}"] = float(record[key])
                except (TypeError, ValueError):
                    pass
        if record.get("optb88vdw_total_energy") is not None:
            atoms.info["JARVIS_optb88vdw_total_energy"] = float(record["optb88vdw_total_energy"])
        before = set(atoms.info) | set(atoms.arrays)
        set_ref_energy_forces_stress(
            atoms,
            energy=record.get("REF_energy"),
            forces=record.get("forces") or record.get("REF_forces"),
            stress=record.get("stress") or record.get("REF_stress"),
        )
        response_labels = set_field_labels(
            atoms,
            electric_field=record.get("REF_electric_field"),
            polarization=record.get("REF_polarization"),
            becs=record.get("REF_becs"),
            polarizability=record.get("REF_polarizability"),
        )
        after = set(atoms.info) | set(atoms.arrays)
        for label, key in (("energy", "REF_energy"), ("forces", "REF_forces"), ("stress", "REF_stress")):
            counts[label] += int(key in after and key not in before)
        for label in ("polarization", "becs", "polarizability"):
            counts[label] += int(response_labels[label])
        frames.append(atoms)

    if not frames:
        raise RuntimeError(f"No usable structures found in JARVIS {dataset}")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="JarvisDB",
        output=output,
        frames=written,
        labels=counts,
        sources=[{**source, "archive": str(archive_path), "member": member, "records": len(records)}],
        notes=[
            "JARVIS dielectric scalars are retained as JARVIS_* metadata; the archive does not contain complete MACEField BEC/polarizability labels.",
            "The published optb88vdw_total_energy field is retained as metadata until its per-atom/total convention is selected for training.",
        ],
    )
    write_json(MANIFEST_DIR / f"{output.stem}.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(ARCHIVES), default="dft_3d")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "JarvisDB.xyz")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(args.output, dataset=args.dataset, max_records=args.max_records)
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
