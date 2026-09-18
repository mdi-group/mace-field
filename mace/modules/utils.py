###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from scipy.constants import c, e

from mace.tools import to_numpy
from mace.tools.scatter import scatter_mean, scatter_std, scatter_sum
from mace.tools.torch_geometric.batch import Batch

from .blocks import AtomicEnergiesBlock


def safe_double(t: torch.Tensor) -> torch.Tensor:
    """Cast to float64 for accumulation precision, except on MPS.

    The Apple-Silicon MPS backend does not support float64, so there the
    tensor is returned unchanged in its working dtype.
    """
    if t.device.type == "mps":
        return t
    return t.double()


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def compute_forces_virials_polarization(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    electric_field: torch.Tensor,
    cell: torch.Tensor,
    create_graph: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute energy first derivatives for forces, stress, and polarization.

    These quantities are all first derivatives of the same scalar energy.  A
    single reverse-mode VJP is therefore cheaper than separately differentiating
    the energy for forces/virials and for polarization.  ``create_graph`` must
    remain enabled when BECs or polarizability will be differentiated from the
    returned polarization.
    """
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials, field_gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions, displacement, electric_field],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )

    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros_like(displacement)
    if field_gradient is None:
        field_gradient = torch.zeros_like(electric_field)

    stress = torch.zeros_like(displacement)
    if compute_stress:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))

    return -forces, -virials, stress, -field_gradient


def compute_forces_polarization(
    energy: torch.Tensor,
    positions: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute force and polarization from one energy reverse-mode pass."""
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, field_gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions, electric_field],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )

    if forces is None:
        forces = torch.zeros_like(positions)
    if field_gradient is None:
        field_gradient = torch.zeros_like(electric_field)
    return -forces, -field_gradient


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
    displacement: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    if displacement is None:
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        displacement = displacement + positions.sum() * 0.0
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def compute_forces_virials_magforces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    magmoms: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:

    # forces correct static type
    grad_outputs = torch.jit.annotate(
        List[Optional[torch.Tensor]], [torch.ones_like(energy)]
    )

    # Pack all inputs into a list
    inputs = [positions, displacement, magmoms]

    grads = torch.autograd.grad(
        outputs=[energy],
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )

    # Explicit unwrapping of Optionals for torch compile
    forces_opt = grads[0]
    virials_opt = grads[1]
    mag_forces_opt = grads[2]

    forces = forces_opt if forces_opt is not None else torch.zeros_like(positions)
    virials = virials_opt if virials_opt is not None else torch.zeros_like(displacement)
    mag_forces = (
        mag_forces_opt if mag_forces_opt is not None else torch.zeros_like(magmoms)
    )

    # Compute stress if requested
    stress = torch.zeros_like(displacement)
    if compute_stress:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))

    return -forces, -virials, stress, -mag_forces


def compute_forces_magforces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    magmoms: torch.Tensor,
    training: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute atomic forces and magnetic forces in a single autograd pass.

    Returns:
        -forces: dE/d(positions)
        -mag_forces: dE/d(magmoms), or None if magmoms not provided
    """

    # forces correct static type
    grad_outputs = torch.jit.annotate(
        List[Optional[torch.Tensor]], [torch.ones_like(energy)]
    )

    inputs = [positions, magmoms]
    grads = torch.autograd.grad(
        outputs=[energy],
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )

    # Explicitly unwrap Optionals so TorchScript knows they are Tensors
    forces_opt = grads[0]
    mag_forces_opt = grads[1]

    forces = forces_opt if forces_opt is not None else torch.zeros_like(positions)
    mag_forces = (
        mag_forces_opt if mag_forces_opt is not None else torch.zeros_like(magmoms)
    )

    return -forces, -mag_forces


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    magmoms: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
    compute_magforces: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:

    if (
        (compute_virials or compute_stress) and displacement is not None
    ) and compute_magforces:
        if magmoms is None:
            raise ValueError("Magnetic moment must be provided to get magnetic forces")
        forces, virials, stress, mag_forces = compute_forces_virials_magforces(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            magmoms=magmoms,
            training=(training or compute_hessian or compute_edge_forces),
            compute_stress=True,
        )
    elif (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian or compute_edge_forces),
        )
        mag_forces = None
    elif compute_force and compute_magforces:
        if magmoms is None:
            raise ValueError("Magnetic moment must be provided to get magnetic forces")
        forces, mag_forces = compute_forces_magforces(
            energy=energy,
            positions=positions,
            magmoms=magmoms,
            training=(training or compute_hessian or compute_edge_forces),
        )
        virials, stress = None, None
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian or compute_edge_forces),
            ),
            None,
            None,
        )
        mag_forces = None
    else:
        forces, virials, stress, mag_forces = (None, None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy,
            positions=vectors,
            training=(training or compute_hessian),
        )
        if edge_forces is not None:
            edge_forces = -1 * edge_forces  # Match LAMMPS sign convention
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces, mag_forces


def get_atomic_virials_stresses(
    edge_forces: torch.Tensor,  # [n_edges, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    vectors: torch.Tensor,  # [n_edges, 3]
    num_atoms: int,
    batch: torch.Tensor,
    cell: torch.Tensor,  # [n_graphs, 3, 3]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute atomic virials and optionally atomic stresses from edge forces and vectors.
    From pobo95 PR #528.
    Returns:
        Tuple of:
            - Atomic virials [num_atoms, 3, 3]
            - Atomic stresses [num_atoms, 3, 3] (None if not computed)
    """
    edge_virial = torch.einsum("zi,zj->zij", edge_forces, vectors)
    atom_virial_sender = scatter_sum(
        src=edge_virial, index=edge_index[0], dim=0, dim_size=num_atoms
    )
    atom_virial_receiver = scatter_sum(
        src=edge_virial, index=edge_index[1], dim=0, dim_size=num_atoms
    )
    atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
    atom_virial = (atom_virial + atom_virial.transpose(-1, -2)) / 2
    atom_stress = None
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    atom_volume = volume[batch].view(-1, 1, 1)
    atom_stress = atom_virial / atom_volume
    atom_stress = torch.where(
        torch.abs(atom_stress) < 1e10, atom_stress, torch.zeros_like(atom_stress)
    )
    return -1 * atom_virial, atom_stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def _check_non_zero(std):
    if np.any(std == 0):
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std[std == 0] = 1
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    out.append(x[:, :num_features])
    for i in range(1, num_layers):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
    return torch.cat(out, dim=-1)


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []
    head_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), batch.head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        head_list.append(batch.head)

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    # mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    # std = to_numpy(torch.std(avg_atom_inter_es)).item()
    mean = to_numpy(scatter_mean(src=avg_atom_inter_es, index=head, dim=0).squeeze(-1))
    std = to_numpy(scatter_std(src=avg_atom_inter_es, index=head, dim=0).squeeze(-1))
    std = _check_non_zero(std)

    return mean, std


def _compute_mean_std_atomic_inter_energy(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head = batch.head
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
    )[torch.arange(batch.num_graphs), head]
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes
    return atom_energies


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    head_list = []
    head_batch = []

    for batch in data_loader:
        head = batch.head
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }
        head_list.append(head)
        head_batch.append(head[batch.batch])

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs]

    # mean = to_numpy(torch.mean(atom_energies)).item()
    # rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )
    rms = _check_non_zero(rms)

    return mean, rms


def _compute_mean_rms_energy_forces(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head = batch.head
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
    )[torch.arange(batch.num_graphs), head]
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes  # {[n_graphs], }
    forces = batch.forces  # {[n_graphs*n_atoms,3], }

    return atom_energies, forces


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_statistics(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []
    head_list = []
    head_batch = []

    for batch in data_loader:
        head = batch.head
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }
        head_list.append(head)  # {[n_graphs], }
        head_batch.append(head[batch.batch])
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs]

    # mean = to_numpy(torch.mean(atom_energies)).item()
    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


def compute_rms_dipoles(
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    dipoles_list = []
    for batch in data_loader:
        dipoles_list.append(batch.dipole)  # {[n_graphs,3], }

    dipoles = torch.cat(dipoles_list, dim=0)  # {[total_n_graphs,3], }
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(dipoles)))).item()
    rms = _check_non_zero(rms)
    return rms


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]


def compute_fixed_charge_dipole_polar(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(
        -1
    )  # / (1e-11 / c / e)  # [N_atoms,3] = 0.20819...
    return scatter_sum(src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)


def compute_total_charge_dipole_permuted(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
):
    dipole_contribution = positions * density_coefficients[:, :1]

    dipole = scatter_sum(
        src=dipole_contribution, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )

    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[..., 1:4], index=batch, dim=-2, dim_size=num_graphs
        )
        dipole = dipole + dipole_p[..., [2, 0, 1]]  # CS phase convention

    total_charge = scatter_sum(
        src=density_coefficients[:, 0], index=batch, dim=-1  # , dim_size=num_graphs
    )

    return total_charge, dipole


@torch.jit.ignore
def compute_dielectric_gradients(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.tensor, torch.tensor]:
    dielectric_flatten = dielectric.view(-1)

    def get_vjp(v):
        return torch.autograd.grad(
            dielectric_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )

    try:
        I_N = torch.eye(dielectric.shape[-1]).to(dielectric.device)
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0)(I_N)[0]
    except RuntimeError:
        gradient = compute_dielectric_gradients_loop(dielectric, positions).detach()
    if gradient is None:
        return torch.zeros((positions.shape[0], dielectric.shape[-1], 3))
    return gradient


def compute_dielectric_gradients_loop(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    gradients = []
    for i in range(dielectric.shape[-1]):
        grad_elem = dielectric[:, i]
        hess_row = torch.autograd.grad(
            grad_elem,
            positions,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        gradients.append(hess_row)
    gradients = torch.stack(gradients)
    return gradients


def get_polarization(
    energy: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            return torch.zeros_like(electric_field)
        energy = energy[graph_mask]
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    polarization = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ...]
        inputs=[electric_field],  # [n_graphs, 3] or [1, 3]
        grad_outputs=grad_outputs,
        # Keep the energy graph alive because force/stress evaluation may have
        # run before this derivative.  ``create_graph`` controls whether the
        # response itself remains differentiable for training.
        retain_graph=True,
        create_graph=create_graph,  # higher derivatives when training
        allow_unused=True,  # <- important
    )[0]

    # If energy does not depend on the field (e.g. foundation model before
    # field-coupling is added), autograd returns None.
    if polarization is None:
        polarization = torch.zeros_like(electric_field)

    # sign convention: P = -∂E/∂E_field
    return -polarization  # [n_graphs, 3]


def get_becs(
    polarization: torch.Tensor,
    positions: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    becs_polar_list = []
    for d in range(3):  # Loop over dimensions
        polar_component = polarization[:, d]  # [n_graphs]
        if graph_mask is not None:
            if not bool(torch.any(graph_mask).item()):
                return torch.zeros(
                    positions.shape[0],
                    3,
                    3,
                    device=positions.device,
                    dtype=positions.dtype,
                )
            polar_component = polar_component[graph_mask]
        polar_grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(polar_component)
        ]
        gradient = torch.autograd.grad(
            outputs=[polar_component],  # [n_graphs]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=polar_grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,  # <- important
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(positions)
        becs_polar_list.append(gradient)  # [n_nodes, 3]
    becs = torch.stack(becs_polar_list, dim=1)  # [n_nodes, 3, 3]
    return becs  # [n_nodes, 3, 3]


def _is_batched_vjp_runtime_error(error: RuntimeError) -> bool:
    """Return whether an eager batched-VJP failure has a safe loop fallback."""
    message = str(error)
    return (
        "Batching rule" in message
        or "vmap" in message
        or "Cannot access data pointer" in message
        or "doesn't have storage" in message
        or "does not have storage" in message
    )


def _get_becs_from_force_field_loop(
    field_gradient: torch.Tensor,
    positions: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BECs through the force--field Maxwell relation.

    For the scalar enthalpy ``E(R, field)`` and MACE's force convention,

    ``dF_beta / dfield_alpha = -d/dR_beta (dE / dfield_alpha)``.

    ``field_gradient`` is ``dE / dfield``.  The mixed derivative is evaluated
    in this reverse-mode ordering because the field has only three components:
    Three field-component VJPs return all atom-coordinate rows without
    materialising a full force-output Jacobian.  This is the Maxwell-equivalent
    ``dF/dfield`` route while keeping the VJP cost proportional to the field
    output.
    """
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            return torch.zeros(
                positions.shape[0],
                3,
                3,
                device=positions.device,
                dtype=positions.dtype,
            )
        field_gradient = field_gradient[graph_mask]

    if not field_gradient.requires_grad:
        return torch.zeros(
            positions.shape[0],
            3,
            3,
            device=positions.device,
            dtype=positions.dtype,
        )

    becs_list: List[torch.Tensor] = []
    for d in range(3):
        field_component = field_gradient[:, d]
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(field_component)
        ]
        gradient = torch.autograd.grad(
            outputs=[field_component],
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(positions)
        becs_list.append(gradient)
    return -torch.stack(becs_list, dim=1)


@torch.jit.ignore
def _get_becs_from_force_field_batched(
    field_gradient: torch.Tensor,
    positions: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager batched-VJP implementation of the Maxwell BEC route."""
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            return torch.zeros(
                positions.shape[0],
                3,
                3,
                device=positions.device,
                dtype=positions.dtype,
            )
        field_gradient = field_gradient[graph_mask]

    if not field_gradient.requires_grad:
        return torch.zeros(
            positions.shape[0],
            3,
            3,
            device=positions.device,
            dtype=positions.dtype,
        )

    vjp_seeds = torch.eye(
        3, device=field_gradient.device, dtype=field_gradient.dtype
    ).unsqueeze(1)
    vjp_seeds = vjp_seeds.expand(3, field_gradient.shape[0], 3)
    try:
        gradients = torch.autograd.grad(
            outputs=[field_gradient],
            inputs=[positions],
            grad_outputs=[vjp_seeds],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
            is_grads_batched=True,
        )[0]
    except RuntimeError as error:
        # Some e3nn/CuEq operations do not yet have batching rules.  Keep the
        # same VJP formulation, but evaluate its three seed directions
        # sequentially rather than failing the whole field model.
        if not _is_batched_vjp_runtime_error(error):
            raise
        return _get_becs_from_force_field_loop(
            field_gradient=field_gradient,
            positions=positions,
            create_graph=create_graph,
            graph_mask=None,
        )
    if gradients is None:
        return torch.zeros(
            positions.shape[0],
            3,
            3,
            device=positions.device,
            dtype=positions.dtype,
        )
    return -gradients.permute(1, 0, 2)


def get_becs_from_force_field(
    field_gradient: torch.Tensor,
    positions: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BECs through the force--field Maxwell relation."""
    if torch.jit.is_scripting():
        return _get_becs_from_force_field_loop(
            field_gradient=field_gradient,
            positions=positions,
            create_graph=create_graph,
            graph_mask=graph_mask,
        )
    return _get_becs_from_force_field_batched(
        field_gradient=field_gradient,
        positions=positions,
        create_graph=create_graph,
        graph_mask=graph_mask,
    )


def _get_becs_and_polarizability_loop(
    polarization: torch.Tensor,
    positions: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BECs and polarizability with memory-bounded VJPs.

    Both quantities are Jacobians of the same three-component polarization.
    Each seed returns gradients with respect to both inputs in one backward
    traversal, so the two Jacobians require three traversals instead of the six
    traversals used when they are evaluated independently.  The seeds are
    processed sequentially to avoid materialising a leading VJP batch dimension
    in the saved autograd intermediates.
    """
    num_graphs = polarization.shape[0]
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            becs_zeros = torch.zeros(
                positions.shape[0],
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
            polarizability_zeros = torch.zeros(
                num_graphs,
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
            return becs_zeros, polarizability_zeros
        polarization = polarization[graph_mask]

    if not polarization.requires_grad:
        becs_zeros = torch.zeros(
            positions.shape[0],
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )
        polarizability_zeros = torch.zeros(
            num_graphs,
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )
        return becs_zeros, polarizability_zeros

    becs_list: List[torch.Tensor] = []
    polarizability_list: List[torch.Tensor] = []
    for d in range(3):
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(polarization[:, d])
        ]
        gradients = torch.autograd.grad(
            outputs=[polarization[:, d]],
            inputs=[positions, electric_field],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        position_gradient = gradients[0]
        field_gradient = gradients[1]
        if position_gradient is None:
            position_gradient = torch.zeros_like(positions)
        if field_gradient is None:
            field_gradient = torch.zeros_like(electric_field)
        becs_list.append(position_gradient)
        polarizability_list.append(field_gradient)

    becs = torch.stack(becs_list, dim=1)
    polarizability = torch.stack(polarizability_list, dim=1)
    return becs, polarizability


@torch.jit.ignore
def _get_becs_and_polarizability_batched(
    polarization: torch.Tensor,
    positions: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute both response Jacobians with batched VJPs for inference."""
    num_graphs = polarization.shape[0]
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            becs_zeros = torch.zeros(
                positions.shape[0],
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
            polarizability_zeros = torch.zeros(
                num_graphs,
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
            return becs_zeros, polarizability_zeros
        polarization = polarization[graph_mask]

    if not polarization.requires_grad:
        becs_zeros = torch.zeros(
            positions.shape[0],
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )
        polarizability_zeros = torch.zeros(
            num_graphs,
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )
        return becs_zeros, polarizability_zeros

    vjp_seeds = torch.eye(
        3, device=polarization.device, dtype=polarization.dtype
    ).unsqueeze(1)
    vjp_seeds = vjp_seeds.expand(3, polarization.shape[0], 3)
    try:
        gradients = torch.autograd.grad(
            outputs=[polarization],
            inputs=[positions, electric_field],
            grad_outputs=[vjp_seeds],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
            is_grads_batched=True,
        )
    except RuntimeError as error:
        if not _is_batched_vjp_runtime_error(error):
            raise
        return _get_becs_and_polarizability_loop(
            polarization=polarization,
            positions=positions,
            electric_field=electric_field,
            create_graph=create_graph,
            graph_mask=None,
        )

    position_gradients = gradients[0]
    field_gradients = gradients[1]
    if position_gradients is None:
        position_gradients = torch.zeros(
            3,
            positions.shape[0],
            3,
            device=positions.device,
            dtype=positions.dtype,
        )
    if field_gradients is None:
        field_gradients = torch.zeros(
            3,
            electric_field.shape[0],
            3,
            device=electric_field.device,
            dtype=electric_field.dtype,
        )
    return position_gradients.permute(1, 0, 2), field_gradients.permute(1, 0, 2)


def get_becs_and_polarizability(
    polarization: torch.Tensor,
    positions: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BECs and polarizability from one shared response VJP pass."""
    if torch.jit.is_scripting() or create_graph:
        return _get_becs_and_polarizability_loop(
            polarization=polarization,
            positions=positions,
            electric_field=electric_field,
            create_graph=create_graph,
            graph_mask=graph_mask,
        )
    return _get_becs_and_polarizability_batched(
        polarization=polarization,
        positions=positions,
        electric_field=electric_field,
        create_graph=create_graph,
        graph_mask=graph_mask,
    )


def _get_polarizability_loop(
    polarization: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_graphs = polarization.shape[0]
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            return torch.zeros(
                num_graphs,
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
        polarization = polarization[graph_mask]

    if not polarization.requires_grad:
        return torch.zeros(
            num_graphs,
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )

    polarizability_list: List[torch.Tensor] = []
    for d in range(3):
        polar_component = polarization[:, d]
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(polar_component)
        ]
        grad_field = torch.autograd.grad(
            outputs=[polar_component],
            inputs=[electric_field],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if grad_field is None:
            grad_field = torch.zeros_like(electric_field)
        polarizability_list.append(grad_field)
    return torch.stack(polarizability_list, dim=1)


@torch.jit.ignore
def _get_polarizability_batched(
    polarization: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager batched-VJP implementation of the polarizability Jacobian."""
    num_graphs = polarization.shape[0]
    if graph_mask is not None:
        if not bool(torch.any(graph_mask).item()):
            return torch.zeros(
                num_graphs,
                3,
                3,
                device=polarization.device,
                dtype=polarization.dtype,
            )
        polarization = polarization[graph_mask]

    if not polarization.requires_grad:
        return torch.zeros(
            num_graphs,
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )

    vjp_seeds = torch.eye(
        3, device=polarization.device, dtype=polarization.dtype
    ).unsqueeze(1)
    vjp_seeds = vjp_seeds.expand(3, polarization.shape[0], 3)
    try:
        gradients = torch.autograd.grad(
            outputs=[polarization],
            inputs=[electric_field],
            grad_outputs=[vjp_seeds],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
            is_grads_batched=True,
        )[0]
    except RuntimeError as error:
        if not _is_batched_vjp_runtime_error(error):
            raise
        return _get_polarizability_loop(
            polarization=polarization,
            electric_field=electric_field,
            create_graph=create_graph,
            graph_mask=None,
        )
    if gradients is None:
        return torch.zeros(
            num_graphs,
            3,
            3,
            device=polarization.device,
            dtype=polarization.dtype,
        )
    return gradients.permute(1, 0, 2)  # [n_graphs, output, field]


def get_polarizability(
    polarization: torch.Tensor,
    electric_field: torch.Tensor,
    create_graph: bool = True,
    graph_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute polarizability with eager batched VJPs and a script fallback."""
    if torch.jit.is_scripting():
        return _get_polarizability_loop(
            polarization=polarization,
            electric_field=electric_field,
            create_graph=create_graph,
            graph_mask=graph_mask,
        )
    return _get_polarizability_batched(
        polarization=polarization,
        electric_field=electric_field,
        create_graph=create_graph,
        graph_mask=graph_mask,
    )


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[torch.Tensor]
    lammps_natoms: Tuple[int, int] = (0, 0)


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    vectors: torch.Tensor
    lengths: torch.Tensor
    cell: torch.Tensor
    node_heads: torch.Tensor
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
) -> GraphContext:
    if torch.jit.is_scripting():
        lammps_mliap = False

    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    )

    if lammps_mliap:
        n_real, n_ghost = data["natoms"][0], data["natoms"][1]
        num_graphs = 2
        num_atoms_arange = torch.arange(n_real, device=data["node_attrs"].device)
        displacement = None
        positions = torch.zeros(
            (int(n_real), 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        cell = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        vectors = data["vectors"].requires_grad_(True)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        ikw = InteractionKwargs(data["lammps_class"], (n_real, n_ghost))
    else:
        if not torch.compiler.is_compiling():
            data["positions"].requires_grad_(True)
        positions = data["positions"]
        cell = data["cell"]
        num_atoms_arange = torch.arange(positions.shape[0], device=positions.device)
        num_graphs = int(data["ptr"].numel() - 1)
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        if compute_virials or compute_stress or compute_displacement:
            p, s, displacement = get_symmetric_displacement(
                positions=positions,
                unit_shifts=data["unit_shifts"],
                cell=cell,
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
                displacement=data.get("displacement"),
            )
            data["positions"], data["shifts"] = p, s
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        is_lammps=lammps_mliap,
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )
