"""Clean and scientifically validate the MACEField foundation datasets.

The collectors normalize labels at ingestion time.  This second pass is kept
separate so raw downloads remain reproducible while training consumes a
conservative, auditable subset.  It checks canonical shapes/finite values,
extreme numerical values, and the acoustic sum rule (ASR) for Born effective
charges.  It also records which response labels co-occur with E/F/stress on
the same structure; matching labels are never inferred across structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import DATA_DIR, MANIFEST_DIR, ensure_workspace, read_frames, write_frames, write_json


LABELS = ("energy", "forces", "stress", "polarization", "becs", "polarizability")
POLARIZATION_KEYS = ("REF_polarization", "REF_total_polarisation", "REF_total_polarization")
REQUIRED_LABELS_BY_DATASET = {
    "MP-Dielectric.xyz": ("becs", "polarizability"),
}
UNITS = {
    "energy": "eV per structure",
    "forces": "eV/Angstrom",
    "stress": "eV/Angstrom^3, ASE convention",
    "electric_field": "V/Angstrom",
    "polarization": "e/Angstrom^2",
    "becs": "e, atom x Cartesian electric direction x Cartesian displacement direction",
    "polarizability": "dimensionless susceptibility epsilon_r - I",
}


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array


def _labels(atoms) -> dict[str, np.ndarray | None]:
    return {
        "energy": _array(atoms.info.get("REF_energy")),
        "forces": _array(atoms.arrays.get("REF_forces")),
        "stress": _array(atoms.info.get("REF_stress")),
        "polarization": next(
            (array for array in (_array(atoms.info.get(key)) for key in POLARIZATION_KEYS) if array is not None),
            None,
        ),
        "becs": _array(atoms.arrays.get("REF_becs")),
        "polarizability": _array(atoms.info.get("REF_polarizability")),
    }


def _norms(values: dict[str, np.ndarray | None], natoms: int) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, value in values.items():
        if value is None:
            result[label] = None
            continue
        if label == "energy":
            result[label] = float(np.abs(value.reshape(-1)[0])) if value.size == 1 else None
        elif label == "forces":
            result[label] = float(np.sqrt(np.mean(value**2))) if value.shape == (natoms, 3) else None
        elif label == "becs":
            result[label] = float(np.sqrt(np.mean(value**2))) if value.shape in ((natoms, 3, 3), (natoms, 9)) else None
        elif label == "stress":
            result[label] = float(np.linalg.norm(value.reshape(-1))) if value.size in (6, 9) else None
        elif label == "polarization":
            result[label] = float(np.linalg.norm(value)) if value.shape == (3,) else None
        else:
            result[label] = float(np.linalg.norm(value.reshape(-1))) if value.size == 9 else None
    return result


def _check_frame(
    atoms,
    *,
    asr_tolerance: float,
    max_energy_per_atom: float,
    max_force: float,
    max_stress: float,
    max_bec: float,
    max_polarization: float,
    max_polarizability: float,
    max_response_atoms: int,
    required_labels: tuple[str, ...] = (),
) -> tuple[list[str], float | None, dict[str, np.ndarray | None]]:
    values = _labels(atoms)
    reasons: list[str] = []
    if required_labels and not any(values[label] is not None for label in required_labels):
        reasons.append("missing_required_response_labels")
    if len(atoms) > max_response_atoms and (
        values["becs"] is not None or values["polarizability"] is not None
    ):
        # Training BECs/polarizabilities requires differentiable second
        # derivatives.  The MACE-MH-1-sized model does not fit these graphs for
        # very large structures on a 24 GB GPU, even at batch size one.
        reasons.append("response_structure_too_large")
    if not np.all(np.isfinite(atoms.positions)):
        reasons.append("positions_nonfinite")
    if atoms.cell.volume <= 0 or not np.isfinite(atoms.cell.volume):
        reasons.append("invalid_cell")
    for label, value in values.items():
        if value is None:
            if any(
                key in atoms.info or key in atoms.arrays
                for key in {
                    "energy": ("REF_energy",),
                    "forces": ("REF_forces",),
                    "stress": ("REF_stress",),
                    "polarization": POLARIZATION_KEYS,
                    "becs": ("REF_becs",),
                    "polarizability": ("REF_polarizability",),
                }[label]
            ):
                reasons.append(f"{label}_not_numeric")
            continue
        if not np.all(np.isfinite(value)):
            reasons.append(f"{label}_nonfinite")
        if label == "energy" and value.size != 1:
            reasons.append("energy_shape")
        elif label == "forces" and value.shape != (len(atoms), 3):
            reasons.append("forces_shape")
        elif label == "stress" and value.size not in (6, 9):
            reasons.append("stress_shape")
        elif label == "polarization" and value.shape != (3,):
            reasons.append("polarization_shape")
        elif label == "becs" and value.shape not in ((len(atoms), 3, 3), (len(atoms), 9)):
            reasons.append("becs_shape")
        elif label == "polarizability" and value.size != 9:
            reasons.append("polarizability_shape")

    norms = _norms(values, len(atoms))
    if norms["energy"] is not None and norms["energy"] / max(len(atoms), 1) > max_energy_per_atom:
        reasons.append("energy_extreme")
    if norms["forces"] is not None and np.max(np.abs(values["forces"])) > max_force:  # type: ignore[arg-type]
        reasons.append("forces_extreme")
    if values["stress"] is not None and np.max(np.abs(values["stress"])) > max_stress:  # type: ignore[arg-type]
        reasons.append("stress_extreme")
    if values["becs"] is not None and np.max(np.abs(values["becs"])) > max_bec:  # type: ignore[arg-type]
        reasons.append("becs_extreme")
    if norms["polarization"] is not None and norms["polarization"] > max_polarization:
        reasons.append("polarization_extreme")
    if norms["polarizability"] is not None and norms["polarizability"] > max_polarizability:
        reasons.append("polarizability_extreme")

    asr_residual = None
    becs = values["becs"]
    if becs is not None and not reasons:
        tensor = becs.reshape(len(atoms), 3, 3)
        asr_residual = float(np.max(np.abs(np.sum(tensor, axis=0))))
        if asr_residual > asr_tolerance:
            reasons.append("becs_acoustic_sum_rule")
    return reasons, asr_residual, values


def _correlations(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    output = {label: {other: None for other in LABELS} for label in LABELS}
    for i, label in enumerate(LABELS):
        for other in LABELS[i + 1 :]:
            pairs = [(row[label], row[other]) for row in rows if row[label] is not None and row[other] is not None]
            if len(pairs) < 3:
                continue
            left, right = np.asarray(pairs, dtype=float).T
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            coefficient = float(np.corrcoef(left, right)[0, 1])
            output[label][other] = coefficient
            output[other][label] = coefficient
    return output


def clean_dataset(
    path: Path,
    output: Path,
    *,
    asr_tolerance: float,
    max_energy_per_atom: float,
    max_force: float,
    max_stress: float,
    max_bec: float,
    max_polarization: float,
    max_polarizability: float,
    max_response_atoms: int,
    required_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    frames = read_frames(path)
    kept = []
    dropped = Counter()
    cooccurrence = {label: {other: 0 for other in LABELS} for label in LABELS}
    norm_rows = []
    asr_values_seen = []
    asr_values_kept = []
    for index, atoms in enumerate(frames):
        reasons, asr_residual, values = _check_frame(
            atoms,
            asr_tolerance=asr_tolerance,
            max_energy_per_atom=max_energy_per_atom,
            max_force=max_force,
            max_stress=max_stress,
            max_bec=max_bec,
            max_polarization=max_polarization,
            max_polarizability=max_polarizability,
            max_response_atoms=max_response_atoms,
            required_labels=required_labels,
        )
        if asr_residual is not None:
            asr_values_seen.append(asr_residual)
        present = [label for label, value in values.items() if value is not None]
        if reasons:
            for reason in reasons:
                dropped[reason] += 1
            continue
        if asr_residual is not None:
            asr_values_kept.append(asr_residual)
        for label in present:
            for other in present:
                cooccurrence[label][other] += 1
            # Some historical extxyz sources carry the target but omit the
            # optional per-configuration weight.  Keep explicit zero weights,
            # but activate an otherwise unweighted real label in the audited
            # copy (including REF_total_polarisation).
            atoms.info.setdefault(f"config_{label}_weight", 1.0)
        kept.append(atoms)
        norm_rows.append(_norms(values, len(atoms)))
    if not kept:
        raise RuntimeError(f"All frames were rejected from {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = write_frames(output, kept)
    return {
        "input": str(path),
        "output": str(output),
        "frames_input": len(frames),
        "frames_output": written,
        "frames_dropped": len(frames) - written,
        "dropped_reasons": dict(sorted(dropped.items())),
        "labels_output": {
            label: sum(1 for row in norm_rows if row[label] is not None) for label in LABELS
        },
        "label_cooccurrence_output": cooccurrence,
        "norm_correlations_output": _correlations(norm_rows),
        "bec_asr_max_abs_seen": max(asr_values_seen) if asr_values_seen else None,
        "bec_asr_p99_abs_seen": float(np.percentile(asr_values_seen, 99)) if asr_values_seen else None,
        "bec_asr_max_abs_kept": max(asr_values_kept) if asr_values_kept else None,
        "bec_asr_p99_abs_kept": float(np.percentile(asr_values_kept, 99)) if asr_values_kept else None,
        "units": UNITS,
        "notes": [
            "Labels are retained only when finite and shape-valid; no labels are fabricated or matched across structures.",
            (
                "At least one valid label is required from: "
                + ", ".join(required_labels)
                + "."
                if required_labels
                else "No dataset-specific response-label requirement was applied."
            ),
            f"BEC acoustic sum rule threshold: max absolute atom-sum <= {asr_tolerance} e.",
            f"Response-labeled structures over {max_response_atoms} atoms are excluded from the training copy to fit second derivatives in 24 GB GPU memory.",
            "Energy/force/stress response matches are counted only when present on the same extxyz frame.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "cleaned")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "cleaned_summary.json")
    parser.add_argument("--asr-tolerance", type=float, default=0.25)
    parser.add_argument("--max-energy-per-atom", type=float, default=1.0e4)
    parser.add_argument("--max-force", type=float, default=1.0e3)
    parser.add_argument("--max-stress", type=float, default=1.0e3)
    parser.add_argument("--max-bec", type=float, default=100.0)
    parser.add_argument("--max-polarization", type=float, default=1.0e3)
    parser.add_argument("--max-polarizability", type=float, default=1.0e6)
    parser.add_argument("--max-response-atoms", type=int, default=128)
    args = parser.parse_args()
    ensure_workspace()
    paths = args.paths or [
        DATA_DIR / "MP-Dielectric.xyz",
        DATA_DIR / "MP-ferroelectric.xyz",
        DATA_DIR / "finite-field-ferroelectric.extxyz",
        DATA_DIR / "JarvisDB.xyz",
        DATA_DIR / "C2DB.xyz",
        DATA_DIR / "Togo.xyz",
    ]
    summaries = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing dataset: {path}")
        summaries.append(
            clean_dataset(
                path,
                args.output_dir / path.name,
                asr_tolerance=args.asr_tolerance,
                max_energy_per_atom=args.max_energy_per_atom,
                max_force=args.max_force,
                max_stress=args.max_stress,
                max_bec=args.max_bec,
                max_polarization=args.max_polarization,
                max_polarizability=args.max_polarizability,
                max_response_atoms=args.max_response_atoms,
                required_labels=REQUIRED_LABELS_BY_DATASET.get(path.name, ()),
            )
        )
    write_json(args.manifest, {"datasets": summaries, "units": UNITS})
    dropped = sum(summary["frames_dropped"] for summary in summaries)
    print(f"Cleaned {len(summaries)} datasets; dropped={dropped}; summary={args.manifest}")


if __name__ == "__main__":
    main()
