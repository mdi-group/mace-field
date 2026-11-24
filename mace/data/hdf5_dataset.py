from pathlib import Path
from typing import List, Union

import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools.utils import AtomicNumberTable


class HDF5Dataset(Dataset):
    def __init__(
        self, file_path, r_max, z_table, atomic_dataclass=AtomicData, **kwargs
    ):
        super(HDF5Dataset, self).__init__()  # pylint: disable=super-with-arguments
        self.file_path = file_path
        self._file = None
        batch_key = list(self.file.keys())[0]
        self._direct_atomic_data = batch_key.startswith(
            "config_"
        ) and not batch_key.startswith("config_batch_")
        if self._direct_atomic_data:
            self.batch_size = 1
            self.length = len(self.file.keys())
        else:
            self.batch_size = len(self.file[batch_key].keys())
            self.length = len(self.file.keys()) * self.batch_size
        self.r_max = r_max
        self.z_table = z_table
        self.atomic_dataclass = atomic_dataclass
        try:
            self.drop_last = bool(self.file.attrs["drop_last"])
        except KeyError:
            self.drop_last = False
        self.kwargs = kwargs

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self._direct_atomic_data:
            return self._get_direct_atomic_data(index)

        # compute the index of the batch
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        grp = self.file["config_batch_" + str(batch_index)]
        subgrp = grp["config_" + str(config_index)]

        properties = {}
        property_weights = {}
        for key in subgrp["properties"]:
            properties[key] = unpack_value(subgrp["properties"][key][()])
        for key in subgrp["property_weights"]:
            property_weights[key] = unpack_value(subgrp["property_weights"][key][()])

        # Configuration.head defaults to "Default", so pass the head here
        # instead of relying on the None fallback below.  Grouped HDF5 files
        # written before the head field was serialized inherit the dataset's
        # configured head, which is the useful behavior for single-head files.
        if "head" in subgrp:
            head = unpack_value(subgrp["head"][()])
        else:
            head = self.kwargs.get("head", "Default")

        config = Configuration(
            atomic_numbers=subgrp["atomic_numbers"][()],
            positions=subgrp["positions"][()],
            properties=properties,
            weight=unpack_value(subgrp["weight"][()]),
            property_weights=property_weights,
            config_type=unpack_value(subgrp["config_type"][()]),
            pbc=unpack_value(subgrp["pbc"][()]),
            cell=unpack_value(subgrp["cell"][()]),
            head=head,
        )
        if config.head is None:
            config.head = self.kwargs.get("head")
        atomic_data = self.atomic_dataclass.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.r_max,
            heads=self.kwargs.get("heads", ["Default"]),
            **{k: v for k, v in self.kwargs.items() if k != "heads"},
        )
        return atomic_data

    def iter_property_with_weight(self, property_name):
        """Yield raw property and weight pairs without building atomic graphs.

        This is used for dataset-wide metadata such as robust loss scales. It
        supports both the grouped configuration schema and the direct
        ``AtomicData`` schema used by older HDF5 exports.
        """

        def read(group, name):
            if name not in group:
                return None
            return unpack_value(group[name][()])

        with h5py.File(self.file_path, "r") as h5_file:
            if self._direct_atomic_data:
                for config_name in h5_file:
                    group = h5_file[config_name]
                    yield (
                        read(group, property_name),
                        read(group, f"{property_name}_weight"),
                    )
                return

            for batch_name in h5_file:
                batch = h5_file[batch_name]
                for config_name in batch:
                    config = batch[config_name]
                    properties = config.get("properties")
                    property_weights = config.get("property_weights")
                    yield (
                        (
                            read(properties, property_name)
                            if properties is not None
                            else None
                        ),
                        (
                            read(property_weights, property_name)
                            if property_weights is not None
                            else None
                        ),
                    )

    def _get_direct_atomic_data(self, index):
        """Read the flat ``save_dataset_as_HDF5`` representation."""
        group = self.file[f"config_{index}"]

        def read(name, default=None):
            if name not in group:
                return default
            return torch.as_tensor(group[name][()])

        hessian = read("hessian")
        hessian_weight = read("hessian_weight")
        if hessian_weight is None:
            hessian_weight = torch.tensor(
                1.0 if hessian is not None else 0.0,
                dtype=torch.get_default_dtype(),
            )

        return self.atomic_dataclass(
            edge_index=read("edge_index"),
            node_attrs=read("node_attrs"),
            positions=read("positions"),
            shifts=read("shifts"),
            unit_shifts=read("unit_shifts"),
            cell=read("cell"),
            weight=read("weight"),
            head=read("head"),
            energy_weight=read("energy_weight"),
            forces_weight=read("forces_weight"),
            stress_weight=read("stress_weight"),
            virials_weight=read("virials_weight"),
            dipole_weight=None,
            charges_weight=None,
            polarization_weight=read("polarization_weight"),
            becs_weight=read("becs_weight"),
            polarizability_weight=read("polarizability_weight"),
            magforces_weight=None,
            forces=read("forces"),
            energy=read("energy"),
            hessian=hessian,
            hessian_weight=hessian_weight,
            stress=read("stress"),
            virials=read("virials"),
            dipole=read("dipole"),
            charges=read("charges"),
            polarizability=read("polarizability"),
            magmom=None,
            magforces=None,
            elec_temp=None,
            pbc=read("pbc"),
            electric_field=read("electric_field"),
            polarization=read("polarization"),
            becs=read("becs"),
        )


def dataset_from_sharded_hdf5(
    files: Union[str, Path], z_table: AtomicNumberTable, r_max: float, **kwargs
):
    directory = Path(files)
    files = sorted(
        str(path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in (".h5", ".hdf5")
    )
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 shard files found in {directory!r}")
    datasets = []
    for file in files:
        datasets.append(HDF5Dataset(file, z_table=z_table, r_max=r_max, **kwargs))
    full_dataset = ConcatDataset(datasets)
    return full_dataset


def configurations_for_e0_estimation(
    files: Union[str, Path, List[Union[str, Path]]], head: str = "Default"
):
    """Yield light grouped-HDF5 configurations for foundation E0 estimation.

    Only geometry and energy are loaded. In particular, large response and
    Hessian arrays are not materialized while estimating per-element energy
    offsets.
    """
    if isinstance(files, (str, Path)):
        files = [files]

    shards = []
    for item in files:
        path = Path(item)
        if path.is_dir():
            shards.extend(
                sorted(
                    shard
                    for shard in path.iterdir()
                    if shard.is_file() and shard.suffix.lower() in (".h5", ".hdf5")
                )
            )
        elif path.is_file() and path.suffix.lower() in (".h5", ".hdf5"):
            shards.append(path)
        else:
            raise ValueError(
                f"E0 estimation expected grouped HDF5 files or directories, got {path}"
            )

    if not shards:
        raise FileNotFoundError(f"No HDF5 shards found in {files!r}")

    for shard in shards:
        with h5py.File(shard, "r") as h5_file:
            for batch in h5_file.values():
                for subgroup in batch.values():
                    properties = subgroup.get("properties")
                    if properties is None or "energy" not in properties:
                        continue
                    energy = unpack_value(properties["energy"][()])
                    if energy is None:
                        continue
                    weights = subgroup.get("property_weights")
                    energy_weight = (
                        unpack_value(weights["energy"][()])
                        if weights is not None and "energy" in weights
                        else 1.0
                    )
                    yield Configuration(
                        atomic_numbers=subgroup["atomic_numbers"][()],
                        positions=subgroup["positions"][()],
                        properties={"energy": energy},
                        property_weights={"energy": energy_weight},
                        cell=unpack_value(subgroup["cell"][()]),
                        pbc=unpack_value(subgroup["pbc"][()]),
                        weight=1.0,
                        head=head,
                    )


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value
