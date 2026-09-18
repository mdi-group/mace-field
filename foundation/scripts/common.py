"""Shared helpers for the local MACE-Field foundation-data workspace.

The foundation directory is intentionally a self-contained, untracked data
workspace.  These helpers keep downloads resumable and keep provenance in
sidecar manifests rather than putting large source payloads in git.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import ase.io
import numpy as np
import requests
from ase import Atoms

FOUNDATION_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = FOUNDATION_ROOT / "data"
MANIFEST_DIR = FOUNDATION_ROOT / "manifests"
CACHE_DIR = FOUNDATION_ROOT / ".cache"

# 1 microC/cm^2 expressed in e/Angstrom^2.
MICROCOULOMB_PER_CM2_TO_E_PER_A2 = 1.0e-2 / 16.02176634


def ensure_workspace() -> None:
    for directory in (DATA_DIR, MANIFEST_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError(f"Cannot encode {type(value)!r} as JSON")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (20.0, 120.0),
    force: bool = False,
) -> Path:
    """Stream *url* to an atomically-renamed cache file."""

    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with client.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def as_numpy(value: Any, dtype: Any = float) -> np.ndarray:
    """Decode common API/list/ndarray encodings into an ndarray."""

    if isinstance(value, Mapping) and "__ndarray__" in value:
        raw = value["__ndarray__"]
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            shape, type_name, flat = raw
            return np.asarray(flat, dtype=np.dtype(type_name)).reshape(shape)
        return np.asarray(raw, dtype=dtype)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=dtype)


def finite_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = as_numpy(value)
    except (TypeError, ValueError):
        return None
    if shape is not None and tuple(array.shape) != shape:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def model_dump(document: Any) -> dict[str, Any]:
    if isinstance(document, Mapping):
        return dict(document)
    if hasattr(document, "model_dump"):
        return document.model_dump(mode="json", by_alias=True)
    if hasattr(document, "dict"):
        return document.dict(by_alias=True)
    raise TypeError(f"Unsupported document type: {type(document)!r}")


def structure_to_atoms(structure: Any) -> Atoms:
    """Convert a pymatgen structure or serialized structure to ASE."""

    from pymatgen.core import Structure

    if isinstance(structure, Structure):
        pymatgen_structure = structure
    elif isinstance(structure, Mapping):
        pymatgen_structure = Structure.from_dict(dict(structure))
    else:
        pymatgen_structure = Structure.from_dict(model_dump(structure))
    return Atoms(
        symbols=[site.specie.symbol for site in pymatgen_structure.sites],
        scaled_positions=np.asarray(pymatgen_structure.frac_coords),
        cell=np.asarray(pymatgen_structure.lattice.matrix, dtype=float),
        pbc=True,
    )


def read_frames(path: Path) -> list[Atoms]:
    frames = ase.io.read(path, index=":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def write_frames(path: Path, frames: Iterable[Atoms]) -> int:
    materialized = list(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        raise ValueError(f"Refusing to write an empty dataset: {path}")
    # extxyz stores per-atom arrays as N or N x 3 columns.  MACEField accepts
    # both (N, 3, 3) and (N, 9) BECs, so flatten only at the file boundary.
    serializable = []
    for atoms in materialized:
        copy = atoms.copy()
        if "REF_becs" in copy.arrays and copy.arrays["REF_becs"].ndim == 3:
            copy.arrays["REF_becs"] = copy.arrays["REF_becs"].reshape(len(copy), 9)
        serializable.append(copy)
    ase.io.write(path, serializable, format="extxyz")
    return len(materialized)


def put_info(atoms: Atoms, key: str, value: Any) -> None:
    """Store a scalar/vector metadata value safely in an ASE frame."""

    if isinstance(value, np.ndarray):
        atoms.info[key] = value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        atoms.info[key] = value.item()
    else:
        atoms.info[key] = value


def set_ref_energy_forces_stress(
    atoms: Atoms,
    *,
    energy: Any = None,
    forces: Any = None,
    stress: Any = None,
) -> None:
    energy_array = finite_array(energy)
    if energy_array is not None and energy_array.size == 1:
        atoms.info["REF_energy"] = float(energy_array.reshape(-1)[0])
        atoms.info["config_energy_weight"] = 1.0
    force_array = finite_array(forces)
    if force_array is not None and force_array.shape == (len(atoms), 3):
        atoms.arrays["REF_forces"] = force_array
        atoms.info["config_forces_weight"] = 1.0
    stress_array = finite_array(stress)
    if stress_array is not None and stress_array.size in (6, 9):
        atoms.info["REF_stress"] = stress_array.reshape(-1).tolist()
        atoms.info["config_stress_weight"] = 1.0


def set_field_labels(
    atoms: Atoms,
    *,
    electric_field: Any = None,
    polarization: Any = None,
    becs: Any = None,
    polarizability: Any = None,
) -> dict[str, bool]:
    """Set only shape-valid MACE-Field labels and activate their weights."""

    result = {"electric_field": False, "polarization": False, "becs": False, "polarizability": False}
    field = finite_array(electric_field, (3,))
    if field is not None:
        atoms.info["REF_electric_field"] = field.tolist()
        result["electric_field"] = True
    polarization_array = finite_array(polarization, (3,))
    if polarization_array is not None:
        atoms.info["REF_polarization"] = polarization_array.tolist()
        atoms.info["config_polarization_weight"] = 1.0
        result["polarization"] = True
    bec_array = finite_array(becs)
    if bec_array is not None and bec_array.shape in ((len(atoms), 3, 3), (len(atoms), 9)):
        atoms.arrays["REF_becs"] = bec_array
        atoms.info["config_becs_weight"] = 1.0
        result["becs"] = True
    alpha_array = finite_array(polarizability)
    if alpha_array is not None and alpha_array.size == 9:
        atoms.info["REF_polarizability"] = alpha_array.reshape(9).tolist()
        atoms.info["config_polarizability_weight"] = 1.0
        result["polarizability"] = True
    return result


def merge_label_results(base: dict[str, bool], extra: Mapping[str, bool]) -> dict[str, bool]:
    """Merge label-presence flags without losing a label from an earlier pass."""

    for key, value in extra.items():
        base[key] = bool(base.get(key, False) or value)
    return base


def manifest_for(
    *,
    dataset: str,
    output: Path,
    sources: list[dict[str, Any]],
    frames: int,
    labels: Mapping[str, int],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "output": str(output),
        "frames": frames,
        "sha256": sha256(output),
        "labels": dict(labels),
        "sources": sources,
        "notes": notes or [],
    }
    return manifest
