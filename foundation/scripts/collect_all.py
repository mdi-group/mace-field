"""Run the foundation-data collectors with resumable source caches."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from common import DATA_DIR, MANIFEST_DIR, ensure_workspace, manifest_for, write_json

LOG = logging.getLogger("collect_all")


def _copy_finite_field() -> None:
    import ase.io

    source = Path(__file__).resolve().parents[2] / "data" / "finite-field-ferroelectric.extxyz"
    output = DATA_DIR / "finite-field-ferroelectric.extxyz"
    shutil.copy2(source, output)
    frames = ase.io.read(output, index=":")
    manifest = manifest_for(
        dataset="finite-field-ferroelectric",
        output=output,
        frames=len(frames),
        labels={},
        sources=[{"path": str(source)}],
        notes=["Copied without changing source frames; field key normalization is a separate step."],
    )
    write_json(MANIFEST_DIR / "finite-field-ferroelectric.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-c2db", type=int, default=None)
    parser.add_argument("--max-togo", type=int, default=None)
    parser.add_argument("--max-jarvis", type=int, default=None)
    parser.add_argument(
        "--jarvis-summary",
        action="store_true",
        help="Use the lightweight JARVIS dft_3d summary archive instead of raw DFPT archives",
    )
    parser.add_argument("--c2db-workers", type=int, default=8)
    parser.add_argument("--togo-workers", type=int, default=4)
    parser.add_argument(
        "--togo-html-index",
        action="store_true",
        help="Use slow MDR HTML pagination rather than the maintained PhononDB index",
    )
    parser.add_argument("--jarvis-workers", type=int, default=4)
    parser.add_argument("--skip-mp", action="store_true")
    parser.add_argument("--skip-mpcontribs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    ensure_workspace()

    jobs = []
    if not args.skip_mp:
        from collect_mp import collect

        jobs.append(("MP-Dielectric", lambda: collect(DATA_DIR / "MP-Dielectric.xyz", chunks=None, dielectric_as_polarizability=True)))
    if not args.skip_mpcontribs:
        from collect_mp_contribs import collect

        jobs.append(("MP-ferroelectric", lambda: collect(DATA_DIR / "MP-ferroelectric.xyz", project="ferroelectrics", chunk_size=100)))
    from collect_c2db import collect as collect_c2db
    from collect_togo import collect as collect_togo

    if args.jarvis_summary:
        from collect_jarvis import collect as collect_jarvis

        jarvis_job = lambda: collect_jarvis(
            DATA_DIR / "JarvisDB.xyz", dataset="dft_3d", max_records=args.max_jarvis
        )
    else:
        from collect_jarvis_dfpt import collect as collect_jarvis

        jarvis_job = lambda: collect_jarvis(
            DATA_DIR / "JarvisDB.xyz", max_archives=args.max_jarvis, workers=args.jarvis_workers
        )

    jobs.extend(
        [
            ("C2DB", lambda: collect_c2db(DATA_DIR / "C2DB.xyz", max_structures=args.max_c2db, workers=args.c2db_workers)),
            ("JarvisDB", jarvis_job),
            (
                "Togo",
                lambda: collect_togo(
                    DATA_DIR / "Togo.xyz",
                    max_records=args.max_togo,
                    workers=args.togo_workers,
                    fast_index=not args.togo_html_index,
                ),
            ),
        ]
    )
    failures = []
    for name, job in jobs:
        try:
            LOG.warning("Collecting %s", name)
            job()
        except Exception as error:  # noqa: BLE001 - report all source failures together.
            failures.append((name, str(error)))
            LOG.error("%s failed: %s", name, error)
            if not args.continue_on_error:
                raise
    try:
        _copy_finite_field()
    except Exception as error:  # noqa: BLE001
        failures.append(("finite-field-ferroelectric", str(error)))
        if not args.continue_on_error:
            raise
    if failures:
        print("Collection completed with failures:")
        for name, error in failures:
            print(f"- {name}: {error}")
        raise SystemExit(1)
    print("All requested foundation datasets collected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
