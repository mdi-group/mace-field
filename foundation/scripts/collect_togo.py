"""Collect phonon structures from the NIMS MDR Togo phonon collection."""

from __future__ import annotations

import argparse
import json
import logging
import lzma
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from ase import Atoms
from bs4 import BeautifulSoup

from common import (
    CACHE_DIR,
    DATA_DIR,
    MANIFEST_DIR,
    as_numpy,
    download_file,
    ensure_workspace,
    manifest_for,
    merge_label_results,
    set_field_labels,
    set_ref_energy_forces_stress,
    write_json,
    write_frames,
)

LOG = logging.getLogger("collect_togo")
BASE_URL = "https://mdr.nims.go.jp"
COLLECTION = "d7aab932-8512-4b9a-b93d-b61f6e5e7019"
PHONONDB_INDEX_URL = "https://raw.githubusercontent.com/atztogo/phonondb/main/mdr/phonondb/README.md"
PAGE_SIZE = 10
TOTAL_RECORDS = 10034


def _get_with_retries(session: requests.Session, url: str, *, params=None, timeout=(15, 90), attempts=5):
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 16))
    raise last_error  # type: ignore[misc]


def _parse_dataset_page(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    page_items = []
    seen = set()
    for link in soup.select('a[href^="/datasets/"]'):
        match = re.fullmatch(r"/datasets/([0-9a-f-]{36})", link.get("href", ""))
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            title = _title(link.get_text(" ", strip=True))
            page_items.append((match.group(1), title))
    return page_items


def _title(value: str) -> str:
    return value if value.startswith("Ab-initio phonon calculation") else ""


def _dataset_page(
    page: int,
    cache: Path,
) -> tuple[int, list[tuple[str, str]]]:
    page_cache = cache / "pages"
    cached_page = page_cache / f"page-{page}.json"
    if cached_page.exists():
        return page, [(identifier, _title(title)) for identifier, title in json.loads(cached_page.read_text(encoding="utf-8"))]
    with requests.Session() as session:
        response = _get_with_retries(
            session,
            f"{BASE_URL}/datasets",
            params={"collection": COLLECTION, "page": page, "locale": "en"},
            timeout=(15, 90),
        )
    page_items = _parse_dataset_page(response.text)
    page_cache.mkdir(parents=True, exist_ok=True)
    cached_page.write_text(json.dumps(page_items), encoding="utf-8")
    return page, page_items


def _dataset_ids(
    session: requests.Session,
    max_records: int | None,
    cache: Path,
    workers: int,
) -> list[tuple[str, str]]:
    del session  # Each concurrent request owns its own Session.
    requested = TOTAL_RECORDS if max_records is None else min(max_records, TOTAL_RECORDS)
    page_count = (requested + PAGE_SIZE - 1) // PAGE_SIZE
    pages: dict[int, list[tuple[str, str]]] = {}
    uncached = []
    page_cache = cache / "pages"
    for page in range(1, page_count + 1):
        if (page_cache / f"page-{page}.json").exists():
            continue
        uncached.append(page)
    if uncached:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(_dataset_page, page, cache): page for page in uncached}
            for future in as_completed(futures):
                page, items = future.result()
                pages[page] = items
    for page in range(1, page_count + 1):
        if page not in pages:
            _, pages[page] = _dataset_page(page, cache)
    result = []
    for page in range(1, page_count + 1):
        result.extend(pages[page])
        if len(result) >= requested:
            break
    if max_records is not None:
        return result[:max_records]
    return result[:TOTAL_RECORDS]


def _phonondb_ids(session: requests.Session, cache: Path) -> list[tuple[str, str]]:
    """Read the maintained 10,034-entry PhononDB index from GitHub."""

    index_path = cache / "phonondb-index.md"
    if index_path.exists():
        markdown = index_path.read_text(encoding="utf-8")
    else:
        response = _get_with_retries(session, PHONONDB_INDEX_URL, timeout=(15, 60))
        markdown = response.text
        cache.mkdir(parents=True, exist_ok=True)
        index_path.write_text(markdown, encoding="utf-8")
    datasets = []
    pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|.*?download_all/([A-Za-z0-9]+)\.zip",
    )
    for line in markdown.splitlines():
        match = pattern.search(line)
        if match:
            mp_id, name, identifier = match.groups()
            datasets.append((identifier, f"PhononDB {name.strip()} (MP-{mp_id})"))
    if not datasets:
        raise RuntimeError(f"No dataset links found in {PHONONDB_INDEX_URL}")
    return datasets


def _yaml_payload(archive: zipfile.ZipFile) -> dict[str, Any] | None:
    candidates = [name for name in archive.namelist() if "phonopy_params.yaml" in name]
    if not candidates:
        candidates = [name for name in archive.namelist() if name.endswith(('.yaml', '.yaml.xz'))]
    if not candidates:
        return None
    import yaml

    name = candidates[0]
    raw = archive.read(name)
    if name.endswith(".xz"):
        raw = lzma.decompress(raw)
    payload = yaml.safe_load(raw)
    return payload if isinstance(payload, dict) else None


def _structure(payload: dict[str, Any]) -> Atoms | None:
    cell = payload.get("unit_cell") or payload.get("unitcell") or payload.get("primitive_cell") or payload
    lattice = payload.get("lattice") or cell.get("lattice")
    points = payload.get("points") or cell.get("points")
    if lattice is None or points is None:
        return None
    symbols = []
    coordinates = []
    for point in points:
        if not isinstance(point, dict):
            continue
        symbol = point.get("symbol") or point.get("element")
        coordinate = point.get("coordinates") or point.get("position")
        if symbol is not None and coordinate is not None:
            symbols.append(symbol)
            coordinates.append(coordinate)
    if not symbols:
        return None
    return Atoms(symbols=symbols, scaled_positions=np.asarray(coordinates, dtype=float), cell=np.asarray(lattice, dtype=float), pbc=True)


def _collect_one(
    identifier: str,
    title: str,
    cache: Path,
    fast_index: bool,
) -> tuple[Atoms | None, dict[str, bool]]:
    archive_cache = cache / "archives" if fast_index else cache
    zip_path = archive_cache / f"{identifier}.zip"
    try:
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            url = (
                f"{BASE_URL}/download_all/{identifier}.zip"
                if fast_index
                else f"{BASE_URL}/datasets/{identifier}.zip"
            )
            last_error = None
            for attempt in range(3):
                try:
                    download_file(
                        url,
                        zip_path,
                        timeout=(15, 180),
                    )
                    break
                except requests.RequestException as error:
                    last_error = error
                    if attempt < 2:
                        time.sleep(2**attempt)
            else:
                raise last_error  # type: ignore[misc]
        raw = zip_path.read_bytes()
    except (OSError, requests.RequestException) as error:
        LOG.warning("Skipping Togo %s: download failed: %s", identifier, error)
        return None, {"becs": False, "polarization": False, "polarizability": False}
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            payload = _yaml_payload(archive)
    except (OSError, zipfile.BadZipFile, lzma.LZMAError) as error:
        LOG.warning("Skipping Togo %s: invalid archive: %s", identifier, error)
        return None, {"becs": False, "polarization": False, "polarizability": False}
    if payload is None:
        return None, {"becs": False, "polarization": False, "polarizability": False}
    atoms = _structure(payload)
    if atoms is None:
        return None, {"becs": False, "polarization": False, "polarizability": False}
    atoms.info["source"] = "NIMS MDR Togo phonon calculation database"
    atoms.info["source_dataset_id"] = identifier
    if title:
        atoms.info["source_title"] = title
    set_ref_energy_forces_stress(
        atoms,
        energy=payload.get("energy"),
        forces=payload.get("forces"),
        stress=payload.get("stress"),
    )
    born = payload.get("born") or payload.get("born_charges") or payload.get("born_effective_charge")
    dielectric = payload.get("dielectric") or payload.get("epsilon") or payload.get("dielectric_constant")
    dielectric_array = None
    if dielectric is not None:
        try:
            dielectric_array = as_numpy(dielectric)
            atoms.info["Togo_dielectric"] = dielectric_array.reshape(-1).tolist()
            atoms.info["Togo_dielectric_units"] = "source phonopy units"
        except (TypeError, ValueError):
            dielectric_array = None
    labels = set_field_labels(atoms, electric_field=[0.0, 0.0, 0.0], becs=born)
    if dielectric_array is not None and dielectric_array.shape == (3, 3):
        # phonopy's dielectric_constant is a relative dielectric tensor; the
        # MACEField output is the corresponding susceptibility epsilon_r - I.
        merge_label_results(
            labels,
            set_field_labels(
                atoms, electric_field=[0.0, 0.0, 0.0], polarizability=dielectric_array - np.eye(3)
            ),
        )
    # Togo is a harmonic response source rather than an E/F/stress source.
    # Keep only structures that actually carry a field-response label; an
    # unlabeled frame would contribute no supervised signal to this head.
    if not any(labels.get(label, False) for label in ("becs", "polarization", "polarizability")):
        return None, labels
    return atoms, labels


def collect(output: Path, *, max_records: int | None, workers: int, fast_index: bool) -> dict[str, Any]:
    ensure_workspace()
    session = requests.Session()
    cache = CACHE_DIR / "togo"
    datasets = _phonondb_ids(session, cache) if fast_index else _dataset_ids(session, max_records, cache, workers)
    if max_records is not None:
        datasets = datasets[:max_records]
    frames = []
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    # MDR is a public archival service; a small worker count avoids opening a
    # connection for every page/archive at once.
    # YAML decompression/parsing is Python-heavy, so threads serialize most of
    # the useful work behind the GIL.  Processes also let the large cached
    # collection finish without making the network the limiting factor.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_collect_one, identifier, title, cache, fast_index): identifier
            for identifier, title in datasets
        }
        for future in as_completed(futures):
            atoms, labels = future.result()
            if atoms is None:
                continue
            frames.append(atoms)
            for label in ("polarization", "becs", "polarizability"):
                counts[label] += int(labels.get(label, False))
            for label, key in (("energy", "REF_energy"), ("forces", "REF_forces"), ("stress", "REF_stress")):
                counts[label] += int(key in atoms.info or key in atoms.arrays)
    frames.sort(key=lambda atoms: str(atoms.info.get("source_dataset_id", "")))
    if not frames:
        raise RuntimeError("Togo collection returned no parseable phonopy structures")
    written = write_frames(output, frames)
    manifest = manifest_for(
        dataset="Togo",
        output=output,
        frames=written,
        labels=counts,
        sources=[
            {"url": f"{BASE_URL}/collections/{COLLECTION}", "collection": COLLECTION, "datasets_considered": len(datasets)},
            {"url": f"{BASE_URL}/datasets/<uuid>.zip", "description": "per-dataset archive"},
            {"url": PHONONDB_INDEX_URL, "description": "maintained bulk index", "fast_index": fast_index},
        ],
        notes=[
            "The collection is primarily harmonic phonon/force-constant data; response labels are emitted only when complete arrays occur in phonopy_params.yaml.xz.",
            "Complete phonopy relative dielectric tensors are converted to MACEField susceptibility epsilon_r-I and retained as REF_polarizability.",
        ],
    )
    write_json(MANIFEST_DIR / "Togo.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "Togo.xyz")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent archive and page workers")
    parser.add_argument(
        "--html-index",
        action="store_true",
        help="Use slow MDR HTML pagination instead of the maintained PhononDB GitHub index",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    manifest = collect(
        args.output,
        max_records=args.max_records,
        workers=args.workers,
        fast_index=not args.html_index,
    )
    print(f"Wrote {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
