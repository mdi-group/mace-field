from types import SimpleNamespace

import h5py
import numpy as np
import torch

from mace.data import (
    Configuration,
    save_configurations_as_HDF5,
)
from mace.tools.run_train_utils import load_dataset_for_path
from mace.tools import AtomicNumberTable, torch_geometric


def _configuration(n_atoms: int, head: str) -> Configuration:
    n_dof = 3 * n_atoms
    hessian = np.arange(n_dof * n_dof, dtype=np.float32).reshape(n_dof, n_dof)
    positions = np.zeros((n_atoms, 3), dtype=float)
    positions[:, 0] = np.arange(n_atoms) * 1.2
    return Configuration(
        atomic_numbers=np.full(n_atoms, 14, dtype=np.int64),
        positions=positions,
        properties={"hessian": hessian},
        property_weights={
            "energy": 0.0,
            "forces": 0.0,
            "stress": 0.0,
            "virials": 0.0,
            "hessian": 1.0,
        },
        cell=np.zeros((3, 3), dtype=float),
        pbc=(False, False, False),
        head=head,
    )


def test_hdf5_hessians_load_and_batch_with_variable_atom_counts(tmp_path):
    configs = [
        _configuration(2, "MP-Dielectric"),
        _configuration(3, "Pheasy-MP-Phonon-v1.1"),
    ]
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    for index, config in enumerate(configs):
        with h5py.File(shard_dir / f"shard_{index}.h5", "w") as h5_file:
            save_configurations_as_HDF5([config], 0, h5_file)

    pheasy_head = "Pheasy-MP-Phonon-v1.1"
    dataset = load_dataset_for_path(
        str(shard_dir),
        r_max=3.0,
        z_table=AtomicNumberTable([14]),
        heads=["MP-Dielectric", pheasy_head],
        head_config=SimpleNamespace(head_name=pheasy_head),
    )
    first, second = dataset[0], dataset[1]
    assert first.hessian.shape == (2 * 2, 3, 3)
    assert second.hessian.shape == (3 * 3, 3, 3)
    assert first.hessian_weight.item() == 1.0
    assert second.hessian_weight.item() == 1.0
    assert first.head.item() == 0
    assert second.head.item() == 1

    expected = torch.tensor(configs[1].properties["hessian"]).reshape(3, 3, 3, 3)
    expected = expected.permute(0, 2, 1, 3).reshape(3 * 3, 3, 3)
    assert torch.equal(second.hessian, expected)

    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, drop_last=False
    )
    batch = next(iter(loader))
    assert batch.hessian.shape == (2 * 2 + 3 * 3, 3, 3)
    assert batch.hessian_weight.shape == (2,)
    assert batch.head.tolist() == [0, 1]
