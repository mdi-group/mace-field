"""Validate collected extxyz files and write a compact summary manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DATA_DIR, MANIFEST_DIR, ensure_workspace, read_frames, write_json


def validate(path: Path) -> dict:
    frames = read_frames(path)
    counts = {"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0}
    errors = []
    elements = set()
    for index, atoms in enumerate(frames):
        elements.update(int(z) for z in atoms.numbers)
        checks = {
            "energy": atoms.info.get("REF_energy"),
            "forces": atoms.arrays.get("REF_forces"),
            "stress": atoms.info.get("REF_stress"),
            "polarization": atoms.info.get("REF_polarization"),
            "becs": atoms.arrays.get("REF_becs"),
            "polarizability": atoms.info.get("REF_polarizability"),
        }
        for label, value in checks.items():
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(array)):
                errors.append(f"frame {index}: {label} contains non-finite values")
                continue
            if label == "energy" and array.size != 1:
                errors.append(f"frame {index}: energy shape {array.shape}")
                continue
            if label == "forces" and array.shape != (len(atoms), 3):
                errors.append(f"frame {index}: forces shape {array.shape}")
                continue
            if label == "stress" and array.size not in (6, 9):
                errors.append(f"frame {index}: stress shape {array.shape}")
                continue
            if label == "polarization" and array.shape != (3,):
                errors.append(f"frame {index}: polarization shape {array.shape}")
                continue
            if label == "becs" and array.shape not in ((len(atoms), 3, 3), (len(atoms), 9)):
                errors.append(f"frame {index}: becs shape {array.shape}")
                continue
            if label == "polarizability" and array.size != 9:
                errors.append(f"frame {index}: polarizability shape {array.shape}")
                continue
            counts[label] += 1
    return {
        "path": str(path),
        "frames": len(frames),
        "elements": sorted(elements),
        "labels": counts,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=MANIFEST_DIR / "dataset_summary.json")
    args = parser.parse_args()
    ensure_workspace()
    paths = args.paths or sorted(DATA_DIR.glob("*.xyz")) + sorted(DATA_DIR.glob("*.extxyz"))
    summaries = [validate(path) for path in paths if path.exists()]
    if not summaries:
        raise SystemExit("No datasets found to validate")
    write_json(args.output, {"datasets": summaries})
    errors = sum(len(summary["errors"]) for summary in summaries)
    print(f"Validated {len(summaries)} datasets; errors={errors}; summary={args.output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
