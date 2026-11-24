"""Training-path coverage for MACEField force Hessians."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from e3nn import o3

from mace import data
from mace.calculators import MACECalculator
from mace.cli.run_train import run as mace_run
from mace.modules.extensions import MACEField
from mace.modules import UniversalFieldLoss, interaction_classes
from mace.modules.utils import compute_hessians_vmap
from mace.tools.torch_tools import default_dtype
from mace.tools.train import _model_output_kwargs, evaluate, take_step
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.scripts_utils import get_loss_fn
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric


def _small_macefield():
    return MACEField(
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
        atomic_energies=np.zeros((1, 2)),
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
        atomic_inter_scale=np.ones(1),
        atomic_inter_shift=np.zeros(1),
        heads=["Default"],
    )


def _h2_with_hessian():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    # A flat 3N x 3N reference is the documented extxyz input representation.
    atoms.info["REF_hessian"] = np.ones((6, 6), dtype=float).reshape(-1)
    return atoms


def _h3_without_hessian():
    return Atoms(
        "H3",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )


def test_cli_builds_hessian_enabled_universal_field_loss():
    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "hessian-smoke",
            "--model",
            "MACEField",
            "--loss",
            "universal_field",
            "--hessian_weight",
            "0.75",
        ]
    )
    loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)

    assert isinstance(loss_fn, UniversalFieldLoss)
    assert loss_fn.hessian_weight.item() == pytest.approx(0.75)


def test_randomized_hvp_matches_dense_hessian_products():
    matrix = torch.tensor(
        [
            [2.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 4.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.3, 5.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2, 6.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.1, 7.0],
        ],
        dtype=torch.float64,
    )
    positions = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    flat = positions.reshape(-1)
    energy = 0.5 * flat @ matrix @ flat
    forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    probes = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0]],
        dtype=torch.float64,
    )

    projected = compute_hessians_vmap(
        forces, positions, create_graph=False, probes=probes
    )

    torch.testing.assert_close(projected, probes @ matrix)


def test_macefield_cli_completes_hessian_training_epoch(tmp_path):
    """The actual training entry point completes a Hessian-supervised epoch."""
    xyz_path = tmp_path / "cli_hessians.xyz"
    frames = []
    for index in range(4):
        atoms = _h2_with_hessian()
        atoms.positions[1, 0] += 0.02 * index
        atoms.info["REF_hessian"] = (
            np.eye(6, dtype=float) * (1.0 + 0.1 * index)
        ).reshape(-1)
        frames.append(atoms)
    write(xyz_path, frames, format="extxyz")

    args = build_default_arg_parser().parse_args(
        [
            "--name=hessian-cli-smoke",
            "--model=MACEField",
            "--loss=universal_field",
            f"--train_file={xyz_path}",
            "--valid_fraction=0.25",
            "--device=cpu",
            "--default_dtype=float64",
            "--compute_forces=False",
            "--hessian_key=REF_hessian",
            "--hessian_weight=1.0",
            "--energy_weight=0.0",
            "--forces_weight=0.0",
            "--stress_weight=0.0",
            "--polarization_weight=0.0",
            "--becs_weight=0.0",
            "--polarizability_weight=0.0",
            "--batch_size=1",
            "--valid_batch_size=1",
            "--max_num_epochs=1",
            "--seed=7",
            "--num_radial_basis=4",
            "--num_cutoff_basis=4",
            "--max_ell=1",
            "--correlation=2",
            "--num_interactions=2",
            "--radial_MLP=[8,8]",
            "--E0s={1: 0.0}",
            "--multiheads_finetuning=False",
            "--plot=False",
            f"--work_dir={tmp_path}",
            f"--model_dir={tmp_path}",
            f"--checkpoints_dir={tmp_path}",
            f"--log_dir={tmp_path}",
            f"--results_dir={tmp_path}",
        ]
    )
    mace_run(args)

    saved_models = list(tmp_path.glob("hessian-cli-smoke*.model"))
    assert saved_models


def test_macefield_trains_and_validates_force_hessian(tmp_path):
    """Read, batch, differentiate, and score a mixed-size Hessian batch."""
    with default_dtype(torch.float64):
        xyz_path = tmp_path / "hessians.xyz"
        write(xyz_path, [_h2_with_hessian(), _h3_without_hessian()], format="extxyz")
        keys = data.KeySpecification.from_defaults()
        _, configs = data.load_from_xyz(str(xyz_path), key_specification=keys)

        assert configs[0].property_weights["hessian"] == 1.0
        assert configs[0].properties["hessian"].shape == (36,)
        assert configs[1].property_weights["hessian"] == 0.0
        assert configs[1].properties["hessian"] is None

        table = AtomicNumberTable([1, 8])
        examples = [
            data.AtomicData.from_config(config, z_table=table, cutoff=4.0)
            for config in configs
        ]
        hdf5_path = tmp_path / "direct.h5"
        data.save_dataset_as_HDF5(examples, str(hdf5_path))
        hdf5_examples = data.HDF5Dataset(
            str(hdf5_path), r_max=4.0, z_table=table
        )
        assert hdf5_examples[0].hessian.shape == (4, 3, 3)
        assert hdf5_examples[0].hessian_weight.item() == 1.0
        assert hdf5_examples[1].hessian.shape == (0, 3, 3)
        assert hdf5_examples[1].hessian_weight.item() == 0.0

        batch = next(
            iter(torch_geometric.dataloader.DataLoader(examples, batch_size=2))
        )
        assert batch.ptr.tolist() == [0, 2, 5]
        assert batch.hessian.shape == (4, 3, 3)
        assert batch.hessian_weight.tolist() == [1.0, 0.0]

        model = _small_macefield()
        loss_fn = UniversalFieldLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            stress_weight=0.0,
            polarization_weight=0.0,
            becs_weight=0.0,
            polarizability_weight=0.0,
            hessian_weight=1.0,
            huber_delta=1.0,
        )
        output_args = {
            "forces": False,
            "virials": False,
            "stress": False,
            "hessian": True,
        }

        output_kwargs = _model_output_kwargs(
            batch, output_args, training=True, loss_fn=loss_fn
        )
        assert output_kwargs["compute_force"] is True
        assert output_kwargs["compute_hessian"] is True
        prediction = model(batch.to_dict(), **output_kwargs)
        assert prediction["hessian"].shape == (8, 15)
        assert torch.isfinite(prediction["hessian"]).all()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_loss, _ = take_step(
            model=model,
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=None,
            output_args=output_args,
            max_grad_norm=None,
            device=torch.device("cpu"),
        )
        assert torch.isfinite(train_loss)
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in model.parameters()
        )

        valid_loss, metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=torch_geometric.dataloader.DataLoader(
                examples, batch_size=2
            ),
            output_args=output_args,
            device=torch.device("cpu"),
        )
        assert np.isfinite(valid_loss)
        assert np.isfinite(metrics["rmse_hessian"])
        assert metrics["mae_hessian"] is None
        assert metrics["hessian_metric"] == "randomized_hvp_rmse_estimate"
        assert np.isfinite(metrics["rmse_hessian_hvp_estimate"])


def test_macefield_calculator_hessian_matches_force_finite_difference():
    """Calculator Hessians use the negative force-Jacobian convention."""
    with default_dtype(torch.float64):
        model = _small_macefield()
        calculator = MACECalculator(
            models=model,
            model_type="MACEField",
            device="cpu",
            default_dtype="float64",
        )
        delta = 1e-4
        base = _h2_with_hessian()
        base.info.pop("REF_hessian")
        plus = base.copy()
        minus = base.copy()
        plus.positions[0, 0] += delta
        minus.positions[0, 0] -= delta
        base.calc = calculator
        plus.calc = calculator
        minus.calc = calculator

        hessian = calculator.get_hessian(atoms=base)
        forces_plus = plus.get_forces()
        forces_minus = minus.get_forces()
        finite_difference = -(
            forces_plus[0, 0] - forces_minus[0, 0]
        ) / (2 * delta)

    assert np.isfinite(hessian).all()
    assert np.isclose(hessian[0, 0, 0], finite_difference, rtol=2e-3, atol=2e-4)
