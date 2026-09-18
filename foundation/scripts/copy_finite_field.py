"""Copy the repository's finite-field ferroelectric dataset into foundation/data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import DATA_DIR, MANIFEST_DIR, ensure_workspace, manifest_for, write_json


SOURCE = Path(__file__).resolve().parents[2] / "data" / "finite-field-ferroelectric.extxyz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "finite-field-ferroelectric.extxyz")
    args = parser.parse_args()
    ensure_workspace()
    if not args.source.exists():
        raise SystemExit(f"Source dataset does not exist: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    manifest = manifest_for(
        dataset="finite-field-ferroelectric",
        output=args.output,
        frames=0,
        labels={},
        sources=[{"path": str(args.source)}],
        notes=[
            "Copied without changing the source frames.",
            "The source uses REF_ionic_polarisation/REF_electronic_polarisation/REF_total_polarisation; normalize explicitly before MACEField training.",
        ],
    )
    # Fill the frame count without reading/re-writing the source.
    import ase.io

    frames = ase.io.read(args.output, index=":")
    manifest["frames"] = len(frames)
    write_json(MANIFEST_DIR / "finite-field-ferroelectric.json", manifest)
    print(f"Copied {manifest['frames']} frames to {manifest['output']}")


if __name__ == "__main__":
    main()
