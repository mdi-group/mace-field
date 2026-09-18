"""Regression coverage for fine-tuning the current MACE-Field model."""

import json
import numpy as np
import torch
from e3nn import o3

from mace import data
from mace.modules import MACEField as PublicMACEField
from mace.modules import utils as module_utils
from mace.modules import ScaleShiftMACE, interaction_classes
from mace.modules.extensions import MACEField
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.torch_tools import default_dtype
from mace.tools.train import valid_err_log
from mace.tools.utils import AtomicNumberTable, MetricsLogger


def test_macefield_public_exports_same_class():
    """The extension module and package-level export share one class."""
    assert PublicMACEField is MACEField


def test_field_validation_log_records_epoch_loss_and_all_rmses(tmp_path):
    """Validation JSONL keeps the epoch loss and field error metrics."""
    logger = MetricsLogger(str(tmp_path), "metrics")
    metrics = {
        "loss": 1.5,
        "rmse_e_per_atom": 0.1,
        "rmse_f": 0.2,
        "rmse_stress": 0.3,
        "rmse_polarization": 0.4,
        "rmse_becs": 0.5,
        "rmse_polarizability": 0.6,
    }

    valid_err_log(
        valid_loss=1.25,
        eval_metrics=metrics,
        logger=logger,
        log_errors="PerAtomFieldRMSE",
        epoch=7,
        valid_loader_name="Default",
    )

    record = json.loads((tmp_path / "metrics.txt").read_text(encoding="utf-8"))
    assert record["mode"] == "eval"
    assert record["epoch"] == 7
    assert record["valid_loss"] == 1.25
    assert record["rmse_polarization"] == 0.4
    assert record["rmse_becs"] == 0.5
    assert record["rmse_polarizability"] == 0.6


def test_missing_field_labels_have_zero_response_weights():
    """Missing response labels must not trigger fabricated derivative targets."""
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]
        ),
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=(True, True, True),
        properties={},
        property_weights={},
    )
    atomic_data = data.AtomicData.from_config(
        config, z_table=AtomicNumberTable([1, 8]), cutoff=4.0
    )

    assert torch.count_nonzero(atomic_data.polarization_weight) == 0
    assert torch.count_nonzero(atomic_data.becs_weight) == 0
    assert torch.count_nonzero(atomic_data.polarizability_weight) == 0


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


def test_legacy_response_derivatives_match_analytic_field_jacobians():
    """Validate BEC and polarizability through autograd identities only."""
    with default_dtype(torch.float64):
        model = MACEField(
            **_foundation_config(
                ["Default"], atomic_energies=np.zeros((1, 2))
            )
        )
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]
            ),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=(True, True, True),
            properties={},
            property_weights={
                "polarization": 1.0,
                "becs": 1.0,
                "polarizability": 1.0,
            },
        )
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=AtomicNumberTable([1, 8]),
            cutoff=4.0,
            heads=model.heads,
        )
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
        )
        field = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        output = model(
            batch.to_dict(),
            training=True,
            compute_force=True,
            compute_polarization=True,
            compute_becs=True,
            compute_polarizability=True,
            electric_field=field,
        )

        force_field_jacobian = []
        for atom in range(3):
            force_rows = []
            for force_component in range(3):
                force_rows.append(
                    torch.autograd.grad(
                        output["forces"][atom, force_component],
                        field,
                        retain_graph=True,
                        create_graph=True,
                    )[0][0]
                )
            force_field_jacobian.append(torch.stack(force_rows, dim=0))
        force_field_jacobian = torch.stack(force_field_jacobian, dim=0)

        # The implementation returns [atom, field component, coordinate].
        assert torch.allclose(
            force_field_jacobian.transpose(1, 2), output["becs"], atol=1.0e-9
        )

        d_polarization_d_field = []
        for component in range(3):
            d_polarization_d_field.append(
                torch.autograd.grad(
                    output["polarization"][:, component].sum(),
                    field,
                    retain_graph=True,
                    create_graph=True,
                )[0]
            )
        d_polarization_d_field = torch.stack(d_polarization_d_field, dim=1)
        eps0 = 8.8541878128e-12 / 1.602176634e-19 / 1e10
        assert torch.allclose(
            d_polarization_d_field / eps0,
            output["polarizability"],
            atol=1.0e-9,
        )


def test_combined_response_training_path_backpropagates_all_targets():
    """The shared first/second derivative path remains trainable."""
    with default_dtype(torch.float64):
        model = MACEField(
            **_foundation_config(
                ["Default"], atomic_energies=np.zeros((1, 2))
            )
        )
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]
            ),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=(True, True, True),
            properties={},
            property_weights={
                "polarization": 1.0,
                "becs": 1.0,
                "polarizability": 1.0,
            },
        )
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=AtomicNumberTable([1, 8]),
            cutoff=4.0,
            heads=model.heads,
        )
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
        )
        output = model(
            batch.to_dict(),
            training=True,
            compute_force=True,
            compute_stress=True,
            compute_polarization=True,
            compute_becs=True,
            compute_polarizability=True,
            electric_field=torch.zeros((1, 3), dtype=torch.float64),
        )

        loss = sum(
            value.square().mean()
            for key in ("energy", "forces", "stress", "polarization", "becs", "polarizability")
            if (value := output[key]) is not None
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)

    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )


def test_combined_force_response_training_path_backpropagates():
    """Force plus polarization training shares its first energy VJP."""
    with default_dtype(torch.float64):
        model = MACEField(
            **_foundation_config(
                ["Default"], atomic_energies=np.zeros((1, 2))
            )
        )
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]
            ),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=(True, True, True),
            properties={},
            property_weights={"polarization": 1.0},
        )
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=AtomicNumberTable([1, 8]),
            cutoff=4.0,
            heads=model.heads,
        )
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
        )
        output = model(
            batch.to_dict(),
            training=True,
            compute_force=True,
            compute_polarization=True,
            electric_field=torch.zeros((1, 3), dtype=torch.float64),
        )
        loss = output["forces"].square().mean() + output["polarization"].square().mean()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)

    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )


def test_response_vjp_storage_error_falls_back_to_sequential(monkeypatch):
    """Unsupported batched VJPs must use the equivalent loop implementation."""
    positions = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    electric_field = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    polarization = torch.stack(
        [
            (positions[:, component].sum() + electric_field[0, component]).reshape(1)
            for component in range(3)
        ],
        dim=1,
    )
    original_grad = torch.autograd.grad

    def grad_with_storage_error(*args, **kwargs):
        if kwargs.get("is_grads_batched", False):
            raise RuntimeError(
                "Cannot access data pointer of Tensor that doesn't have storage"
            )
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", grad_with_storage_error)
    becs, polarizability = module_utils.get_becs_and_polarizability(
        polarization=polarization,
        positions=positions,
        electric_field=electric_field,
        create_graph=False,
    )

    assert becs.shape == (3, 3, 3)
    assert polarizability.shape == (1, 3, 3)
    assert torch.isfinite(becs).all()
    assert torch.isfinite(polarizability).all()


def test_multhead_pt_head_reproduces_single_head_foundation():
    """The replay head must be an exact zero-field continuation of the source."""
    with default_dtype(torch.float32):
        foundation = ScaleShiftMACE(
            **_foundation_config(
                ["foundation"],
                atomic_energies=np.array([[-2.0, -3.0]]),
            )
        )
        target = MACEField(
            **_foundation_config(
                ["pt_head", "field"],
                atomic_energies=np.array([[-2.0, -3.0], [-20.0, -30.0]]),
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
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.32, 0.94, 0.0]]
            ),
            pbc=(True, True, True),
            cell=np.diag([8.0, 8.0, 8.0]),
            properties={"electric_field": np.zeros(3)},
            property_weights={"electric_field": 0.0},
            head="pt_head",
        )
        source_data = data.AtomicData.from_config(
            config, z_table=table, cutoff=4.0, heads=["foundation"]
        )
        target_data = data.AtomicData.from_config(
            config, z_table=table, cutoff=4.0, heads=target.heads
        )
        source_batch = next(
            iter(torch_geometric.dataloader.DataLoader([source_data], batch_size=1))
        )
        target_batch = next(
            iter(torch_geometric.dataloader.DataLoader([target_data], batch_size=1))
        )
        source_output = foundation(
            source_batch.to_dict(), training=False, compute_force=False
        )
        target_output = target(
            target_batch.to_dict(), training=False, compute_force=False
        )

    assert torch.allclose(
        source_output["energy"], target_output["energy"], atol=1.0e-5, rtol=0.0
    )


def test_field_finetuning_preserves_target_atomic_energy_reference():
    """Field fine-tuning must not restore the foundation checkpoint's E0s."""
    with default_dtype(torch.float32):
        foundation = ScaleShiftMACE(
            **_foundation_config(
                ["Default"],
                atomic_energies=np.array([[-2.0], [-3.0]]),
            )
        )
        target_e0s = np.array([[-20.0], [-30.0]])
        target = MACEField(
            **_foundation_config(["Default"], atomic_energies=target_e0s)
        )

        load_foundations_elements(
            target,
            foundation,
            table=AtomicNumberTable([1, 8]),
            load_readout=True,
            max_L=1,
        )

    assert torch.allclose(
        target.atomic_energies_fn.atomic_energies,
        torch.as_tensor(target_e0s, dtype=torch.float32),
    )


def test_foundation_field_modules_start_small_but_trainable():
    """Foundation transfer must not shock either field architecture."""
    with default_dtype(torch.float32):
        foundation = ScaleShiftMACE(
            **_foundation_config(["Default"], atomic_energies=np.zeros((1, 2)))
        )
        legacy = MACEField(
            **_foundation_config(["Default"], atomic_energies=np.zeros((1, 2)))
        )
        table = AtomicNumberTable([1, 8])
        load_foundations_elements(legacy, foundation, table, load_readout=True, max_L=1)

    legacy_parameters = [
        parameter
        for module_name in ("field_feats", "field_linear")
        for parameter in getattr(legacy, module_name).parameters()
    ]
    assert any(torch.count_nonzero(parameter) for parameter in legacy_parameters)
    assert max(parameter.abs().max().item() for parameter in legacy_parameters) < 1.0e-2
