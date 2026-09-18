"""Audit Materials Project task metadata and raw-file download routes.

The public MP task API exposes task metadata and selected VASP-derived output,
but it is not itself a public OUTCAR file tree.  This report records what the
current API returns for selected task/material identifiers and whether
``MPRester.get_download_info`` can resolve a raw OUTCAR/vasprun.xml download.
It never writes the API key or full task documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import MANIFEST_DIR, ensure_workspace, model_dump, write_json


def _key() -> str:
    import os

    key = os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
    if not key:
        raise SystemExit("Set MP_API_KEY for the MP task audit; it is never written to the report.")
    return key


def _keys(value: Any) -> list[str]:
    return sorted(model_dump(value).keys()) if value is not None else []


def audit(task_ids: list[str], material_ids: list[str], output: Path) -> dict[str, Any]:
    from mp_api.client import MPRester

    ensure_workspace()
    report: dict[str, Any] = {
        "task_ids": task_ids,
        "material_ids": material_ids,
        "task_documents": [],
        "material_summaries": [],
        "raw_file_route": {},
    }
    with MPRester(api_key=_key()) as mpr:
        if task_ids:
            try:
                documents = mpr.materials.tasks.search(
                    task_ids=task_ids,
                    all_fields=True,
                )
                for document in documents:
                    record = model_dump(document)
                    report["task_documents"].append(
                        {
                            "task_id": str(record.get("task_id", "")),
                            "material_id": str(record.get("material_id", "")),
                            "top_level_keys": sorted(record),
                            "input_keys": _keys(record.get("input")),
                            "output_keys": _keys(record.get("output")),
                            "vasp_objects_keys": _keys(record.get("vasp_objects")),
                            "has_structure": record.get("structure") is not None,
                            "has_dir_name": bool(record.get("dir_name")),
                        }
                    )
            except Exception as error:  # noqa: BLE001 - preserve audit evidence.
                report["task_error"] = f"{type(error).__name__}: {error}"
        if material_ids:
            try:
                summaries = mpr.materials.summary.search(
                    material_ids=material_ids,
                    all_fields=True,
                )
                for summary in summaries:
                    record = model_dump(summary)
                    report["material_summaries"].append(
                        {
                            "material_id": str(record.get("material_id", "")),
                            "top_level_keys": sorted(record),
                            "has_structure": record.get("structure") is not None,
                            "has_task_ids": bool(record.get("task_ids")),
                            "has_calc_types": bool(record.get("calc_types")),
                        }
                    )
            except Exception as error:  # noqa: BLE001 - preserve audit evidence.
                report["material_error"] = f"{type(error).__name__}: {error}"
        try:
            info = mpr.get_download_info(
                material_ids or task_ids,
                file_patterns=["OUTCAR", "vasprun.xml"],
            )
            report["raw_file_route"] = {
                "status": "resolved",
                "result_type": type(info).__name__,
                "result_keys": sorted(info) if isinstance(info, dict) else [],
            }
        except Exception as error:  # noqa: BLE001 - preserve route failure.
            report["raw_file_route"] = {
                "status": "unavailable",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", action="append", default=["aaadjozp"])
    parser.add_argument("--material-id", action="append", default=["mp-768203"])
    parser.add_argument("--output", type=Path, default=MANIFEST_DIR / "mp_task_audit.json")
    args = parser.parse_args()
    report = audit(args.task_id, args.material_id, args.output)
    print(f"Wrote MP task audit to {args.output} ({len(report['task_documents'])} task documents)")


if __name__ == "__main__":
    main()
