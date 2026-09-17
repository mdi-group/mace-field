"""Regression coverage for fine-tuning the current MACE-Field model."""

import numpy as np
import torch
from e3nn import o3

from mace import data
from mace.modules import MACEField as PublicMACEField
from mace.modules import ScaleShiftMACE, interaction_classes
from mace.modules.extensions import MACEField
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.torch_tools import default_dtype
from mace.tools.utils import AtomicNumberTable


def test_macefield_public_exports_same_class():
    """The extension module and package-level export share one class."""
    assert PublicMACEField is MACEField


def _foundation_config(heads, atomic_energies):
    return dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.asarray(atomic_energies),
        avg_num_neighbors=4.0,
        atomic_numbers=[1, 8],
        correlation=3,
        radial_type="bessel",
        radial_MLP=[8, 8],
        apply_cutoff=False,
        use_edge_irreps_first=True,
        edge_irreps=o3.Irreps("4x0e + 4x1o"),
        use_agnostic_product=True,
        pair_repulsion=True,
        distance_transform="Agnesi",
        atomic_inter_scale=np.ones(len(heads)),
        atomic_inter_shift=np.zeros(len(heads)),
        heads=heads,
    )


def test_macefield_finetunes_new_multhead_foundation():
    """A nonlinear multi-head foundation can initialize a one-head field model."""
    with default_dtype(torch.float32):
        foundation_heads = ["head_0", "head_1"]
        foundation = ScaleShiftMACE(
            **_foundation_config(
                foundation_heads,
                atomic_energies=np.zeros((len(foundation_heads), 2)),
            )
        )
        target_heads = ["head_1"]
        target = MACEField(
            **_foundation_config(
                target_heads,
                atomic_energies=np.zeros((len(target_heads), 2)),
            )
        )
        table = AtomicNumberTable([1, 8])

        load_foundations_elements(
            target,
            foundation,
            table=table,
            load_readout=True,
            max_L=1,
        )

        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]),
            pbc=(True, True, True),
            cell=np.diag([8.0, 8.0, 8.0]),
            properties={},
            property_weights={},
        )
        atomic_data = data.AtomicData.from_config(
            config, z_table=table, cutoff=4.0, heads=target.heads
        )
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
        )

        output = target(
            batch.to_dict(),
            training=False,
            compute_force=False,
            compute_stress=False,
            compute_polarization=True,
            compute_becs=True,
            compute_polarizability=True,
            electric_field=torch.tensor([0.01, 0.0, 0.0]),
        )

    assert output["energy"].shape == (1,)
    assert output["polarization"].shape == (1, 3)
    assert output["becs"].shape == (3, 3, 3)
    assert output["polarizability"].shape == (1, 3, 3)
    assert all(
        torch.isfinite(value).all() for value in output.values() if value is not None
    )
