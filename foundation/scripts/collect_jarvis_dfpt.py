"""Collect BEC/dielectric data from the JARVIS DFPT raw-output project.

The Figshare project contains one ZIP per JVASP identifier.  Each archive is
downloaded into a resumable cache and parsed from POSCAR/CONTCAR and OUTCAR.
Only records with a complete atom-wise BEC array are emitted.  OUTCAR energy,
force, and stress values are retained when ASE can read them from that same
calculation; no cross-database matching is attempted.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from ase import io as ase_io

from common import (
    CACHE_DIR,
    DATA_DIR,
    MANIFEST_DIR,
    download_file,
    ensure_workspace,
    manifest_for,
    merge_label_results,
    set_field_labels,
    set_ref_energy_forces_stress,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_jarvis_dfpt")
PROJECT_ID = 82118
PROJECT_URL = "https://figshare.com/projects/JARVIS-DFT_DFPT_raw_input_output_files/82118"
API_URL = f"https://api.figshare.com/v2/projects/{PROJECT_ID}/articles"


def _index(cache: Path) -> list[dict[str, Any]]:
    index_path = cache / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    articles = requests.get(API_URL, timeout=60)
    articles.raise_for_status()
    files = []
    for article in articles.json():
        detail = requests.get(article["url"], timeout=60)
        detail.raise_for_status()
        for file in detail.json().get("files", []):
            files.append(
                {
                    "name": file["name"],
                    "download_url": file["download_url"],
                    "size": file["size"],
                    "article": article["id"],
                }
            )
    files.sort(key=lambda item: item["name"])
    cache.mkdir(parents=True, exist_ok=True)
    write_json(index_path, files)
    return files


def _float_tokens(line: str) -> list[float]:
    token_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
    return [float(token.replace("D", "E").replace("d", "e")) for token in re.findall(token_pattern, line)]


def _born_charges(outcar: str, natoms: int) -> np.ndarray | None:
    headers = list(re.finditer(r"POSITION\s+DIRECTION\s+\d+\s+BORN EFFECTIVE CHARGE", outcar))
    if len(headers) < 3:
        return None
    blocks = []
    for header in headers[:3]:
        block = []
        for line in outcar[header.end() :].splitlines():
            if "total drift" in line.lower():
                if "*" in line:
                    return None
                break
            # VASP writes asterisks when a BEC component overflows its output
            # field.  Do not let the numeric suffix of an overflow token be
            # interpreted as a real charge by _float_tokens.
            if "*" in line:
                return None
            values = _float_tokens(line)
            if len(values) >= 6:
                block.append(values[3:6])
            if len(block) == natoms:
                break
        if len(block) != natoms:
            return None
        blocks.append(block)
    return np.asarray(blocks, dtype=float).transpose(1, 0, 2)


def _dielectric(outcar: str) -> np.ndarray | None:
    match = re.search(r"MACROSCOPIC STATIC DIELECTRIC TENSOR.*?\n(.*?)(?:\n\s*-{3,}|\Z)", outcar, re.DOTALL)
    if match is None:
        return None
    rows = []
    for line in match.group(1).splitlines():
        values = _float_tokens(line)
        if len(values) >= 3:
            rows.append(values[:3])
        if len(rows) == 3:
            break
    return np.asarray(rows, dtype=float) if len(rows) == 3 else None


def _read_archive(path: Path, identifier: str):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        structure_name = "CONTCAR" if "CONTCAR" in names else "POSCAR"
        structure_text = archive.read(structure_name).decode(errors="replace")
        outcar = archive.read("OUTCAR").decode(errors="replace") if "OUTCAR" in names else ""
    atoms = ase_io.read(io.StringIO(structure_text), format="vasp")
    atoms.info["source"] = "JARVIS-DFT DFPT raw Figshare archive"
    atoms.info["source_jid"] = identifier
    becs = _born_charges(outcar, len(atoms))
    if becs is None:
        return None
    calculator = None
    try:
        # ASE's OUTCAR parser needs a filesystem path because it resolves
        # auxiliary VASP files relative to ``fd.name``.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".OUTCAR", encoding="utf-8") as handle:
            handle.write(outcar)
            handle.flush()
            calculator = ase_io.read(handle.name, index=-1, format="vasp-out")
    except (OSError, ValueError, IndexError):
        pass
    results = getattr(getattr(calculator, "calc", None), "results", {})
    set_ref_energy_forces_stress(
        atoms,
        energy=results.get("energy"),
        forces=results.get("forces"),
        stress=results.get("stress"),
    )
    dielectric = _dielectric(outcar)
    if dielectric is not None:
        atoms.info["JARVIS_dielectric_static"] = dielectric.reshape(-1).tolist()
        atoms.info["JARVIS_dielectric_units"] = "relative permittivity"
    labels = set_field_labels(atoms, electric_field=[0.0, 0.0, 0.0], becs=becs)
    if dielectric is not None:
        # The MACEField response head returns the susceptibility relative to
        # eps0.  Keep this conversion explicit in the manifest and metadata.
        alpha = dielectric - np.eye(3)
        merge_label_results(
            labels,
            set_field_labels(atoms, electric_field=[0.0, 0.0, 0.0], polarizability=alpha),
        )
    return atoms, labels


def _one(item: dict[str, Any], cache: Path):
    identifier = Path(item["name"]).stem
    destination = cache / "zips" / item["name"]
    last_error = None
    for attempt in range(3):
        try:
            download_file(item["download_url"], destination, timeout=(20, 180))
            return _read_archive(destination, identifier)
        except (OSError, ValueError, zipfile.BadZipFile, requests.RequestException) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    LOG.warning("Skipping %s after retries: %s", identifier, last_error)
    return None


def collect(output: Path, *, max_archives: int | None, workers: int) -> dict[str, Any]:
    ensure_workspace()
    cache = CACHE_DIR / "jarvis_dfpt"
    items = _index(cache)
    if max_archives is not None:
        items = items[:max_archives]
    # Parse/download missing archives first.  The final output still parses
    # every selected archive, but this avoids spending the first several
    # minutes reparsing an existing cache before making progress on the bulk
    # Figshare download.
    items.sort(key=lambda item: ((cache / "zips" / item["name"]).exists(), item["name"]))
    frames = []
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    # OUTCAR parsing is dominated by Python/ASE work and does not scale with a
    # thread pool because of the GIL.  Use processes so a completed Figshare
    # cache can be parsed in a reasonable time on a multi-core host.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, item, cache) for item in items]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            atoms, labels = result
            frames.append(atoms)
            for label in ("polarization", "becs", "polarizability"):
                counts[label] += int(labels.get(label, False))
            for label, key in (("energy", "REF_energy"), ("forces", "REF_forces"), ("stress", "REF_stress")):
                counts[label] += int(key in atoms.info or key in atoms.arrays)
    frames.sort(key=lambda atoms: str(atoms.info.get("source_jid", "")))
    if not frames:
        raise RuntimeError("No complete JARVIS DFPT BEC records were parsed")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="JarvisDB",
        output=output,
        frames=written,
        labels=counts,
        sources=[
            {"url": PROJECT_URL, "project_id": PROJECT_ID, "archives_available": len(_index(cache)), "archives_considered": len(items)},
            {"url": API_URL, "description": "Figshare archive index"},
        ],
        notes=[
            "Complete atom-wise BECs are parsed from the three OUTCAR Born-charge direction blocks.",
            "Energy, forces, and stress are retained only when ASE reads them from the same OUTCAR.",
            "OUTCAR static dielectric tensors are converted to epsilon-I for REF_polarizability; verify this response convention before training.",
            "The cache is resumable and can be continued with a larger --max-archives value.",
        ],
    )
    write_json(MANIFEST_DIR / f"{output.stem}.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "JarvisDB.xyz")
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(args.output, max_archives=args.max_archives, workers=args.workers)
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
