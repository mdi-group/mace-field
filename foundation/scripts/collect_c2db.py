"""Collect response-bearing structures from the public C2DB web service.

C2DB publishes ordinary E/F/stress extxyz downloads and response files in
its public ``downloadable`` tree.  The collector retains BECs only when the
published Born-charge array is complete for the downloaded structure.  C2DB
polarizabilities and spontaneous polarization use 2-D units (Angstrom and
pC/m respectively), so they are stored as explicitly named metadata rather
than silently relabelled as 3-D MACEField targets.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from ase import io
from ase.formula import Formula

from common import (
    CACHE_DIR,
    DATA_DIR,
    MANIFEST_DIR,
    as_numpy,
    ensure_workspace,
    finite_array,
    manifest_for,
    set_field_labels,
    set_ref_energy_forces_stress,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_c2db")
BASE_URL = "https://c2db.fysik.dtu.dk"
RESPONSE_FILTERS = ("alphax>0", "alphax_el>0", "P_spontaneous_norm>0")


def _candidate_uids(session: requests.Session) -> tuple[list[str], dict[str, int]]:
    uids: set[str] = set()
    counts: dict[str, int] = {}
    for filter_text in RESPONSE_FILTERS:
        response = session.get(f"{BASE_URL}/api/query", params={"filter": filter_text}, timeout=60)
        response.raise_for_status()
        counts[filter_text] = int(response.json()["count"])
        landing = session.get(f"{BASE_URL}/", params={"filter": filter_text}, timeout=60)
        landing.raise_for_status()
        sid_match = re.search(r'name="sid"\s+value="([^"]+)"', landing.text)
        if sid_match is None:
            raise RuntimeError(f"C2DB did not return a table session for {filter_text}")
        pages = (counts[filter_text] + 24) // 25
        for page in range(pages):
            table = session.get(
                f"{BASE_URL}/table",
                params={"sid": sid_match.group(1), "page": page},
                timeout=60,
            )
            table.raise_for_status()
            uids.update(re.findall(r'href\s*=\s*["\']?/material/([^"\'\s>]+)', table.text))
    return sorted(uids), counts


def _formula_path(atoms, uid: str) -> str:
    formula = atoms.get_chemical_formula(mode="hill")
    stoichiometry, reduced, nunits = Formula(formula).stoichiometry()
    tag = uid.rsplit("-", 1)[-1]
    return f"materials/{str(stoichiometry).replace(' ', '')}/{nunits}{str(reduced)}/{tag}"


def _first_ndarray(payload: Any, key: str) -> np.ndarray | None:
    if isinstance(payload, dict):
        if key in payload:
            try:
                return as_numpy(payload[key])
            except (TypeError, ValueError):
                return None
        for value in payload.values():
            result = _first_ndarray(value, key)
            if result is not None:
                return result
    elif isinstance(payload, list):
        for value in payload:
            result = _first_ndarray(value, key)
            if result is not None:
                return result
    return None


def _raw_json(session: requests.Session, relative: str, cache: Path) -> Any | None:
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache.unlink(missing_ok=True)
    response = session.get(f"{BASE_URL}/downloadable/{relative}", timeout=60)
    if response.status_code != 200:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(response.text, encoding="utf-8")
    try:
        return response.json()
    except ValueError:
        return None


def _collect_one(uid: str, cache_root: Path) -> tuple[Any | None, dict[str, bool], str | None]:
    session = requests.Session()
    uid_cache = cache_root / uid
    xyz_path = uid_cache.with_suffix(".xyz")
    try:
        if xyz_path.exists():
            atoms = io.read(xyz_path, format="extxyz")
        else:
            response = session.get(f"{BASE_URL}/material/{uid}/download/xyz", timeout=90)
            response.raise_for_status()
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            xyz_path.write_text(response.text, encoding="utf-8")
            atoms = io.read(StringIO(response.text), format="extxyz")
        if isinstance(atoms, list):
            atoms = atoms[0]
        atoms.info["source"] = "C2DB public download"
        atoms.info["source_uid"] = uid
        calculator_results = getattr(getattr(atoms, "calc", None), "results", {})
        set_ref_energy_forces_stress(
            atoms,
            energy=atoms.info.get("energy", calculator_results.get("energy")),
            forces=atoms.arrays.get("forces", calculator_results.get("forces")),
            stress=atoms.info.get("stress", calculator_results.get("stress")),
        )
        relative = _formula_path(atoms, uid)
        born = _raw_json(session, f"{relative}/results-asr.borncharges.json", uid_cache.with_suffix(".born.json"))
        becs = None if born is None else _first_ndarray(born, "Z_avv")
        labels = set_field_labels(atoms, becs=becs)
        if becs is not None and not labels["becs"]:
            atoms.info["C2DB_becs_symmetry_reduced"] = True

        ir = _raw_json(session, f"{relative}/ir-polarizability.json", uid_cache.with_suffix(".ir.json"))
        alpha = None if ir is None else _first_ndarray(ir, "alpha_re_wvv")
        if alpha is not None and alpha.ndim == 3 and alpha.shape[0] > 0:
            zero_index = int(np.argmin(np.abs(np.asarray(ir.get("omega_w", [0.0]), dtype=float)))) if isinstance(ir, dict) else 0
            zero_index = min(zero_index, alpha.shape[0] - 1)
            atoms.info["C2DB_polarizability_Angstrom"] = alpha[zero_index].reshape(-1).tolist()
            atoms.info["C2DB_polarizability_units"] = "Angstrom (2D)"

        spontaneous = _raw_json(
            session,
            f"{relative}/results-asr.spontaneous_polarization.json",
            uid_cache.with_suffix(".spontaneous.json"),
        )
        if spontaneous is not None:
            vector = None
            for key in ("P_v", "P", "polarization"):
                candidate = _first_ndarray(spontaneous, key)
                if candidate is not None and candidate.size == 3:
                    vector = candidate.reshape(3)
                    break
            if vector is not None:
                atoms.info["C2DB_polarization_raw"] = vector.tolist()
                atoms.info["C2DB_polarization_units"] = "pC/m (2D)"
        return atoms, labels, relative
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        LOG.warning("Skipping C2DB %s: %s", uid, error)
        return None, {"becs": False}, None


def collect(output: Path, *, max_structures: int | None, workers: int) -> dict[str, Any]:
    ensure_workspace()
    session = requests.Session()
    uids, candidate_counts = _candidate_uids(session)
    if max_structures is not None:
        uids = uids[:max_structures]
    cache_root = CACHE_DIR / "c2db"
    frames = []
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_collect_one, uid, cache_root): uid for uid in uids}
        for future in as_completed(futures):
            atoms, labels, _ = future.result()
            if atoms is None:
                continue
            frames.append(atoms)
            for label in ("polarization", "becs", "polarizability"):
                counts[label] += int(labels.get(label, False))
            for label, key in (("energy", "REF_energy"), ("forces", "REF_forces"), ("stress", "REF_stress")):
                counts[label] += int(key in atoms.info or key in atoms.arrays)
    frames.sort(key=lambda atoms: str(atoms.info.get("source_uid", "")))
    if not frames:
        raise RuntimeError("C2DB returned no usable structures")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="C2DB",
        output=output,
        frames=written,
        labels=counts,
        sources=[
            {"url": BASE_URL, "candidate_filters": candidate_counts, "candidate_uids": len(uids)},
            {"url": f"{BASE_URL}/material/<uid>/download/xyz", "description": "E/F/stress structure downloads"},
            {"url": f"{BASE_URL}/downloadable/materials/<path>/results-asr.borncharges.json", "description": "Born charges"},
        ],
        notes=[
            "BECs are emitted only for complete atom-wise arrays; symmetry-reduced arrays remain provenance-only.",
            "C2DB static polarizability (Angstrom) and spontaneous polarization (pC/m) are retained as explicitly named 2-D metadata, not MACEField REF targets.",
            "The source uses the live C2DB endpoint because the user-provided URL had a missing .dk suffix.",
        ],
    )
    write_json(MANIFEST_DIR / "C2DB.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "C2DB.xyz")
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(args.output, max_structures=args.max_structures, workers=args.workers)
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
