from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    HEAD = "head"
    CHARGES = "REF_charges"
    FIELD = "REF_electric_field"
    POLARISATION = "REF_polarisation"
    BECS = "REF_bec"
    POLARISABILITY = "REF_polarisability"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict
