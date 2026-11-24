###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch
import torch.distributed as dist

from mace.tools import TensorDict, fold_polarization
from mace.tools.torch_geometric import Batch


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if raw_loss.numel() == 0:
        if ddp and dist.is_initialized():
            total_samples = torch.tensor(
                0, device=raw_loss.device, dtype=raw_loss.dtype
            )
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return torch.tensor(0.0, device=raw_loss.device, dtype=raw_loss.dtype)
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


def polarizability_to_six(polarizability: torch.Tensor) -> torch.Tensor:
    """Return the symmetric six-component representation of a 3x3 tensor.

    The off-diagonal components carry a ``sqrt(2)`` factor so that the
    Euclidean norm of the six-vector is the Frobenius norm of the symmetric
    tensor.  This makes the loss rotationally consistent while avoiding the
    duplicate lower-triangular entries of a matrix target.
    """
    tensor = polarizability.view(-1, 3, 3)
    symmetric = 0.5 * (tensor + tensor.transpose(-1, -2))
    sqrt_two = tensor.new_tensor(2.0).sqrt()
    return torch.stack(
        (
            symmetric[:, 0, 0],
            symmetric[:, 1, 1],
            symmetric[:, 2, 2],
            sqrt_two * symmetric[:, 0, 1],
            sqrt_two * symmetric[:, 0, 2],
            sqrt_two * symmetric[:, 1, 2],
        ),
        dim=-1,
    )


def _weighted_config_mean(
    per_config_loss: torch.Tensor,
    config_weight: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Average one scalar loss per configuration using active config weights."""
    ddp = is_ddp_enabled() if ddp is None else ddp
    config_weight = config_weight.to(dtype=per_config_loss.dtype)
    local_sum = (per_config_loss * config_weight).sum()
    local_weight = config_weight.sum()
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_weight, op=dist.ReduceOp.SUM)
        if local_weight.item() == 0.0:
            return local_sum * 0.0
        return local_sum * world_size / local_weight
    if local_weight.item() == 0.0:
        return local_sum * 0.0
    return local_sum / local_weight


def polarization_quantum_lattice(cell: torch.Tensor) -> torch.Tensor:
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().clamp_min(1e-30).view(-1, 1, 1)
    return cell / volume


def normalized_metric_polarization_distance(
    folded_fractional_polarization: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """
    Return the branch-folded polarization distance measured with the
    cell-aware metric 3 * (Q Q^T) / tr(Q Q^T), where dP = c Q in the
    row-vector convention used here.
    """
    quantum_lattice = polarization_quantum_lattice(cell)
    metric = torch.matmul(quantum_lattice, quantum_lattice.transpose(-2, -1))
    metric_trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(dim=-1)
    metric_trace = metric_trace.clamp_min(1e-30)
    normalized_metric = (3.0 / metric_trace).view(-1, 1, 1) * metric
    return torch.einsum(
        "bi,bij,bj->b",
        folded_fractional_polarization,
        normalized_metric,
        folded_fractional_polarization,
    )


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        magforces_weight=1.0,
        huber_delta=0.01,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "magforces_weight",
            torch.tensor(magforces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        configs_magforces_weight = torch.repeat_interleave(
            ref.magforces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_magforces = reduce_loss(loss_magforces, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="mean",
                    delta=self.huber_delta,
                )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
            + self.magforces_weight * loss_magforces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, magforces_weight={self.magforces_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):
    def __init__(
        self, dipole_weight=1.0, polarizability_weight=1.0
    ) -> (
        None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (
            self.dipole_weight * loss_dipole
            + self.polarizability_weight * loss_polarizability
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class UniversalFieldLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        polarization_weight=1.0,
        becs_weight=1.0,
        polarizability_weight=1.0,
        huber_delta=0.01,
        polarization_loss_mode="normalized_metric",
        polarization_huber_delta=None,
        polarization_loss_scale=1.0,
        polarizability_loss_mode="standardized_symmetric_huber",
        polarizability_huber_delta=1.0,
        polarizability_scales=None,
    ) -> None:
        super().__init__()
        if polarization_loss_mode not in {"cartesian_huber", "normalized_metric"}:
            raise ValueError(
                "polarization_loss_mode must be one of "
                "{'cartesian_huber', 'normalized_metric'}"
            )
        if polarizability_loss_mode not in {
            "raw_huber",
            "standardized_symmetric_huber",
        }:
            raise ValueError(
                "polarizability_loss_mode must be one of "
                "{'raw_huber', 'standardized_symmetric_huber'}"
            )
        if polarizability_huber_delta <= 0:
            raise ValueError("polarizability_huber_delta must be positive")
        self.huber_delta = huber_delta
        self.polarization_loss_mode = polarization_loss_mode
        self.polarization_huber_delta = (
            None
            if polarization_huber_delta is None
            else float(polarization_huber_delta)
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_loss_scale",
            torch.tensor(polarization_loss_scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "becs_weight",
            torch.tensor(becs_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )
        if polarizability_scales is None:
            polarizability_scales = torch.ones(
                (1, 6), dtype=torch.get_default_dtype()
            )
        else:
            polarizability_scales = torch.as_tensor(
                polarizability_scales, dtype=torch.get_default_dtype()
            )
            if polarizability_scales.ndim == 1:
                polarizability_scales = polarizability_scales.view(1, 6)
        if (
            polarizability_scales.ndim != 2
            or polarizability_scales.shape[-1] != 6
            or not torch.isfinite(polarizability_scales).all()
            or torch.any(polarizability_scales <= 0)
        ):
            raise ValueError(
                "polarizability_scales must be a finite positive tensor with "
                "shape [n_heads, 6]"
            )
        self.polarizability_loss_mode = polarizability_loss_mode
        self.polarizability_huber_delta = float(polarizability_huber_delta)
        self.register_buffer("polarizability_scales", polarizability_scales)

    def _polarizability_scales_for_batch(
        self, ref: Batch, num_configs: int
    ) -> torch.Tensor:
        """Select the robust six-component scale for each graph head."""
        if self.polarizability_scales.shape[0] == 1:
            return self.polarizability_scales.expand(num_configs, -1)
        head = getattr(ref, "head", None)
        if head is None:
            head_index = torch.zeros(
                num_configs,
                dtype=torch.long,
                device=self.polarizability_scales.device,
            )
        else:
            head_index = head.view(-1).long()
            if head_index.numel() == 1 and num_configs != 1:
                head_index = head_index.expand(num_configs)
            if head_index.numel() != num_configs:
                raise ValueError(
                    "Batch head index count does not match polarizability labels"
                )
            head_index = head_index.to(self.polarizability_scales.device)
            if torch.any(head_index < 0) or torch.any(
                head_index >= self.polarizability_scales.shape[0]
            ):
                raise ValueError("Batch contains a head outside polarizability_scales")
        return self.polarizability_scales[head_index]

    def _compute_polarizability_loss(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool]
    ) -> torch.Tensor:
        reference = ref["polarizability"].view(-1, 3, 3)
        prediction = pred["polarizability"].view(-1, 3, 3)
        config_weight = ref.polarizability_weight.view(-1, 3, 3).mean(dim=(-1, -2))

        if self.polarizability_loss_mode == "raw_huber":
            element_loss = torch.nn.functional.huber_loss(
                prediction,
                reference,
                reduction="none",
                delta=self.polarizability_huber_delta,
            )
        else:
            reference_six = polarizability_to_six(reference)
            prediction_six = polarizability_to_six(prediction)
            scales = self._polarizability_scales_for_batch(
                ref, reference_six.shape[0]
            ).to(device=reference_six.device, dtype=reference_six.dtype)
            element_loss = torch.nn.functional.huber_loss(
                prediction_six / scales,
                reference_six / scales,
                reduction="none",
                delta=self.polarizability_huber_delta,
            )

        per_config_loss = element_loss.reshape(element_loss.shape[0], -1).mean(dim=-1)
        return _weighted_config_mean(per_config_loss, config_weight, ddp)

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        # Helper: check that a key is present when its batch labels are active
        def _require_key(name: str, weight_buf: torch.Tensor) -> bool:
            """Return True if this loss term should be used; raise if required but missing."""
            ref_weight = getattr(ref, f"{name}_weight", None)
            has_active_ref = ref_weight is not None and bool(
                torch.any(ref_weight != 0).item()
            )
            has_ref = (has_active_ref and (hasattr(ref, name) or (name in ref)))
            has_pred = name in pred

            if float(weight_buf) == 0.0 or not has_active_ref:
                # Skip globally disabled terms and batches without labels.  The
                # latter permits MACEField's response-aware output selection to
                # avoid unnecessary second derivatives on replay-only batches.
                return False

            if not has_ref or not has_pred:
                missing_side = []
                if not has_ref:
                    missing_side.append("ref")
                if not has_pred:
                    missing_side.append("pred")
                raise ValueError(
                    f"UniversalFieldLoss requires '{name}' when its weight is non-zero, "
                    f"but it is missing in {', '.join(missing_side)}."
                )
            return True

        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, num_atoms
        ).unsqueeze(-1)

        use_polarization = _require_key("polarization", self.polarization_weight)
        use_becs = _require_key("becs", self.becs_weight)
        use_polarizability = _require_key("polarizability", self.polarizability_weight)
        polarization_huber_delta = (
            self.huber_delta * 5.0
            if self.polarization_huber_delta is None
            else self.polarization_huber_delta
        )

        if use_polarization:
            configs_polarization_weight = ref.polarization_weight.view(-1, 3)
            # The normalized metric loss is scalar per configuration, so collapse
            # the per-component weights to a single config weight.
            config_polarization_weight = configs_polarization_weight.mean(dim=-1)
            dP_folded, c_folded = fold_polarization(
                pred["polarization"], ref["polarization"], ref["cell"]
            )
        else:
            configs_polarization_weight = None
            config_polarization_weight = None
            dP_folded = None
            c_folded = None

        if use_becs:
            configs_becs_weight = torch.repeat_interleave(
                ref.becs_weight,
                num_atoms,
                dim=0,
            ).view(-1, 3, 3)
        else:
            configs_becs_weight = None

        # --- energy / forces / stress ---
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)

            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )

            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)

            if use_polarization:
                if self.polarization_loss_mode == "normalized_metric":
                    polarization_residual = (
                        config_polarization_weight
                        * normalized_metric_polarization_distance(c_folded, ref["cell"])
                    )
                else:
                    polarization_residual = configs_polarization_weight * (
                        torch.nn.functional.huber_loss(
                            dP_folded,
                            torch.zeros_like(dP_folded),
                            reduction="none",
                            delta=polarization_huber_delta,
                        )
                    )
                loss_polarization = reduce_loss(
                    polarization_residual,
                    ddp,
                )
            else:
                loss_polarization = torch.tensor(
                    0.0, device=configs_energy_weight.device
                )

            if use_becs:
                loss_becs = torch.nn.functional.huber_loss(
                    configs_becs_weight * ref["becs"],
                    configs_becs_weight * pred["becs"],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_becs = reduce_loss(loss_becs, ddp)
            else:
                loss_becs = torch.tensor(0.0, device=configs_energy_weight.device)

            if use_polarizability:
                loss_polarizability = self._compute_polarizability_loss(
                    ref, pred, ddp
                )
            else:
                loss_polarizability = torch.tensor(
                    0.0, device=configs_energy_weight.device
                )
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )

            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )

            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )

            if use_polarization:
                if self.polarization_loss_mode == "normalized_metric":
                    loss_polarization = torch.mean(
                        config_polarization_weight
                        * normalized_metric_polarization_distance(c_folded, ref["cell"])
                    )
                else:
                    polarization_residual = torch.nn.functional.huber_loss(
                        dP_folded,
                        torch.zeros_like(dP_folded),
                        reduction="none",
                        delta=polarization_huber_delta,
                    )
                    loss_polarization = torch.mean(
                        configs_polarization_weight * polarization_residual
                    )
            else:
                loss_polarization = torch.tensor(
                    0.0, device=configs_energy_weight.device
                )

            if use_becs:
                loss_becs = torch.nn.functional.huber_loss(
                    configs_becs_weight * ref["becs"],
                    configs_becs_weight * pred["becs"],
                    reduction="mean",
                    delta=self.huber_delta,
                )
            else:
                loss_becs = torch.tensor(0.0, device=configs_energy_weight.device)

            if use_polarizability:
                loss_polarizability = self._compute_polarizability_loss(
                    ref, pred, ddp
                )
            else:
                loss_polarizability = torch.tensor(
                    0.0, device=configs_energy_weight.device
                )

        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
            + self.polarization_weight
            * self.polarization_loss_scale
            * loss_polarization
            + self.becs_weight * loss_becs
            + self.polarizability_weight * loss_polarizability
        )
