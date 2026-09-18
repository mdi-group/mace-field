from typing import Optional

import torch

from mace.tools.utils import AtomicNumberTable


_MACE_FOUNDATION_CLASSES = {
    "MACE",
    "ScaleShiftMACE",
    "MACELES",
    "PolarMACE",
    "MagneticScaleShiftMACE",
}


def is_mace_foundation_model(model: torch.nn.Module) -> bool:
    """Return whether ``model`` is a plain MACE-family foundation model.

    MACEField-to-MACEField transfer and specialized MDP models must not be
    treated as the plain-MACE-to-MACEField initialization path.  Keep this
    check class-based because the transfer loader receives the unwrapped
    checkpoint module before distributed/compiled wrappers are applied.
    """
    return model.__class__.__name__ in _MACE_FOUNDATION_CLASSES


def _copy_radial_weights(
    model: torch.nn.Module, model_foundations: torch.nn.Module
) -> None:
    """Copy the radial basis weights from the foundation model in-place.

    Preserves the target's buffer-vs-Parameter registration (BesselBasis /
    GaussianBasis expose ``*_weights`` as a Parameter only when ``trainable=True``;
    otherwise it is a register_buffer). Handles both radial classes.
    """
    dst = model.radial_embedding.bessel_fn
    src = model_foundations.radial_embedding.bessel_fn
    attr = {"BesselBasis": "bessel_weights", "GaussianBasis": "gaussian_weights"}.get(
        dst.__class__.__name__
    )
    if attr is None:
        return
    getattr(dst, attr).data.copy_(getattr(src, attr).data)


def _copy_readout_heads(
    model: torch.nn.Module, model_foundations: torch.nn.Module
) -> bool:
    """Transfer scalar readouts while changing the number of heads.

    MACE-MH-1 has six heads and its nonlinear readout stores the hidden
    channels grouped by head.  Repeating the flattened tensor (the old
    loader behavior) corrupts that layout and produces an invalid linear
    weight for an unchanged six-head target or a one-head fine-tune target.
    """
    source_heads = list(getattr(model_foundations, "heads", []))
    target_heads = list(getattr(model, "heads", []))
    if not source_heads or not target_heads:
        return False
    if len(model.readouts) != len(model_foundations.readouts):
        return False
    source_indices = [
        source_heads.index(head) if head in source_heads else 0 for head in target_heads
    ]

    def _irreps_dim(linear: torch.nn.Module) -> Optional[int]:
        irreps_out = getattr(linear, "irreps_out", None)
        if irreps_out is None:
            return None
        return getattr(irreps_out, "dim", getattr(irreps_out, "num_irreps", None))

    for target, source in zip(model.readouts, model_foundations.readouts):
        if target.__class__.__name__ == "LinearReadoutBlock":
            source_input_dim = source.linear.weight.numel() // len(source_heads)
            target_input_dim = target.linear.weight.numel() // len(target_heads)
            if source_input_dim != target_input_dim:
                return False
            source_weight = source.linear.weight.detach().reshape(
                source_input_dim, len(source_heads)
            )
            target.linear.weight = torch.nn.Parameter(
                source_weight[:, source_indices].reshape(-1).clone()
            )
            continue

        if target.__class__.__name__ != "NonLinearReadoutBlock":
            return False
        if not hasattr(target, "linear_1") or not hasattr(target, "linear_2"):
            return False

        source_hidden_dim = _irreps_dim(source.linear_1)
        target_hidden_dim = _irreps_dim(target.linear_1)
        if source_hidden_dim is None or target_hidden_dim is None:
            return False
        if source_hidden_dim % len(source_heads) or target_hidden_dim % len(
            target_heads
        ):
            return False
        source_hidden_per_head = source_hidden_dim // len(source_heads)
        target_hidden_per_head = target_hidden_dim // len(target_heads)
        if source_hidden_per_head != target_hidden_per_head:
            return False
        source_input_dim = source.linear_1.weight.numel() // source_hidden_dim
        target_input_dim = target.linear_1.weight.numel() // target_hidden_dim
        if source_input_dim != target_input_dim:
            return False
        source_linear_1 = source.linear_1.weight.detach().reshape(
            source_input_dim, source_hidden_dim
        )
        target.linear_1.weight = torch.nn.Parameter(
            torch.cat(
                [
                    source_linear_1[
                        :,
                        index
                        * source_hidden_per_head : (index + 1)
                        * source_hidden_per_head,
                    ]
                    for index in source_indices
                ],
                dim=1,
            )
            .reshape(-1)
            .clone()
        )
        if (
            source.linear_1.bias is not None
            and target.linear_1.bias is not None
            and source.linear_1.bias.numel() > 0
            and target.linear_1.bias.numel() > 0
        ):
            source_bias = source.linear_1.bias.detach().reshape(
                len(source_heads), source_hidden_per_head
            )
            target.linear_1.bias = torch.nn.Parameter(
                source_bias[source_indices].reshape(-1).clone()
            )

        source_linear_2_input_dim = source.linear_2.weight.numel() // len(source_heads)
        target_linear_2_input_dim = target.linear_2.weight.numel() // len(target_heads)
        if source_linear_2_input_dim != source_hidden_dim:
            return False
        if target_linear_2_input_dim != target_hidden_dim:
            return False
        source_linear_2 = source.linear_2.weight.detach().reshape(
            source_linear_2_input_dim, len(source_heads)
        )
        target_linear_2 = torch.zeros(
            target_linear_2_input_dim,
            len(target_heads),
            dtype=source_linear_2.dtype,
            device=source_linear_2.device,
        )
        for target_index, source_index in enumerate(source_indices):
            target_linear_2[
                target_index
                * target_hidden_per_head : (target_index + 1)
                * target_hidden_per_head,
                target_index,
            ] = source_linear_2[
                source_index
                * source_hidden_per_head : (source_index + 1)
                * source_hidden_per_head,
                source_index,
            ]
        target.linear_2.weight = torch.nn.Parameter(target_linear_2.reshape(-1).clone())
        # e3nn normalizes a path by the multiplicities of both the input and
        # output irreps.  Expanding a one-head scalar output to N heads thus
        # changes the raw-weight normalization even when each head receives
        # an identical diagonal copy of the source readout.  Compensate using
        # the actual instruction path weights rather than assuming a square
        # root of the head count; this also remains correct for a source model
        # that already has multiple heads.
        source_instructions = getattr(source.linear_2, "instructions", [])
        target_instructions = getattr(target.linear_2, "instructions", [])
        if len(source_instructions) == 1 and len(target_instructions) == 1:
            source_path_weight = source_instructions[0].path_weight
            target_path_weight = target_instructions[0].path_weight
            if target_path_weight != 0.0:
                target.linear_2.weight.data.mul_(
                    source_path_weight / target_path_weight
                )
        if (
            source.linear_2.bias is not None
            and target.linear_2.bias is not None
            and source.linear_2.bias.numel() > 0
            and target.linear_2.bias.numel() > 0
        ):
            target.linear_2.bias = torch.nn.Parameter(
                source.linear_2.bias.detach()[source_indices].clone()
            )
    return True


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
    default_dtype: Optional[torch.dtype] = None,
):
    """Dispatch loader: magnetic models use a specialized skip-tp / magmom-skip-tp
    layout; everything else falls through to the default (vanilla MACE) path.
    ``default_dtype`` is only meaningful for the default branch (the magnetic
    branch was ported from a pre-default_dtype cut of the paper repo)."""
    if "Magnetic" in str(model.__class__.__name__):
        return load_foundations_elements_magnetic(
            model, model_foundations, table, load_readout, use_shift, use_scale, max_L
        )
    return load_foundations_elements_default(
        model,
        model_foundations,
        table,
        load_readout,
        use_shift,
        use_scale,
        max_L,
        default_dtype=default_dtype,
    )


def load_foundations_elements_default(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
    default_dtype: Optional[torch.dtype] = None,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    # Field fine-tuning uses the target dataset's per-atom energy reference.
    # A foundation checkpoint's shift is normally copied for ordinary MACE,
    # but MACEField targets commonly have a different energy zero (and the
    # field extension must not reintroduce the extensive offset bug).
    field_target = hasattr(model, "field_feats")

    def assign_parameter_if_compatible(
        module: torch.nn.Module, name: str, value: torch.Tensor
    ) -> None:
        target = getattr(module, name)
        setattr(module, name, torch.nn.Parameter(value.clone()))

    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    target_dtype = default_dtype or next(model.parameters()).dtype
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    if hasattr(model, "joint_embedding"):
        for (_, param_1), (_, param_2) in zip(
            model.joint_embedding.named_parameters(),
            model_foundations.joint_embedding.named_parameters(),
        ):
            param_1.data.copy_(param_2.data)
    if hasattr(model, "embedding_readout"):
        for (_, param_1), (_, param_2) in zip(
            model.embedding_readout.named_parameters(),
            model_foundations.embedding_readout.named_parameters(),
        ):
            param_1.data.copy_(
                param_2.data.reshape(-1, 1)
                .repeat(1, len(model_heads))
                .flatten()
                .clone()
            )
    _copy_radial_weights(model, model_foundations)
    for i in range(int(model.num_interactions)):
        assign_parameter_if_compatible(
            model.interactions[i].linear_up,
            "weight",
            model_foundations.interactions[i].linear_up.weight,
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(
                    param_2.data[: (num_radial + 2 * num_species), ...]
                )
        if hasattr(model.interactions[i], "linear"):
            assign_parameter_if_compatible(
                model.interactions[i].linear,
                "weight",
                model_foundations.interactions[i].linear.weight,
            )
        if hasattr(model.interactions[i], "linear_1"):
            assign_parameter_if_compatible(
                model.interactions[i].linear_1,
                "weight",
                model_foundations.interactions[i].linear_1.weight,
            )
        if hasattr(model.interactions[i], "linear_2"):
            assign_parameter_if_compatible(
                model.interactions[i].linear_2,
                "weight",
                model_foundations.interactions[i].linear_2.weight,
            )
        if hasattr(model.interactions[i], "linear_res"):
            assign_parameter_if_compatible(
                model.interactions[i].linear_res,
                "weight",
                model_foundations.interactions[i].linear_res.weight,
            )
        if hasattr(model.interactions[i], "source_embedding"):
            source_embedding = (
                model_foundations.interactions[i]
                .source_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
            assign_parameter_if_compatible(
                model.interactions[i].source_embedding,
                "weight",
                source_embedding,
            )
        if hasattr(model.interactions[i], "target_embedding"):
            target_embedding = (
                model_foundations.interactions[i]
                .target_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
            assign_parameter_if_compatible(
                model.interactions[i].target_embedding,
                "weight",
                target_embedding,
            )
        if hasattr(model.interactions[i], "alpha"):
            assign_parameter_if_compatible(
                model.interactions[i],
                "alpha",
                model_foundations.interactions[i].alpha,
            )
        if hasattr(model.interactions[i], "beta"):
            assign_parameter_if_compatible(
                model.interactions[i],
                "beta",
                model_foundations.interactions[i].beta,
            )
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            skip_weight = (
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
            assign_parameter_if_compatible(
                model.interactions[i].skip_tp, "weight", skip_weight
            )
        elif model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            assign_parameter_if_compatible(
                model.interactions[i].skip_tp,
                "weight",
                model_foundations.interactions[i].skip_tp.weight,
            )
        else:
            skip_weight = (
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    (max_ell + 1),
                    num_species_foundations,
                    num_channels_foundation,
                )[:, :, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
            assign_parameter_if_compatible(
                model.interactions[i].skip_tp, "weight", skip_weight
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # Transferring products
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        max_range = max_L + 1 if i < len(model.products) - 1 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            source_weights_max = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights_max[indices_weights_prod, :, :]
                .clone()
            )
            target_contraction = product.symmetric_contractions.contractions[j]
            target_contraction.weights_max = torch.nn.Parameter(source_weights_max)

            target_weights = target_contraction.weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                source_weight = source_weights[k][indices_weights_prod, :, :].clone()
                target_weights[k] = torch.nn.Parameter(source_weight)
        assign_parameter_if_compatible(
            product.linear,
            "weight",
            model_foundations.products[i].linear.weight,
        )

    readouts_loaded = (
        _copy_readout_heads(model, model_foundations) if load_readout else False
    )
    if load_readout and not readouts_loaded:
        # Transferring readouts
        for i, readout in enumerate(model.readouts):
            if readout.__class__.__name__ == "LinearReadoutBlock":
                model_readouts_zero_linear_weight = readout.linear.weight.clone()
                model_readouts_zero_linear_weight = (
                    model_foundations.readouts[i]
                    .linear.weight.view(num_channels_foundation, -1)
                    .repeat(1, len(model_heads))
                    .flatten()
                    .clone()
                )
                readout.linear.weight = torch.nn.Parameter(
                    model_readouts_zero_linear_weight
                )
            if readout.__class__.__name__ in [
                "NonLinearBiasReadoutBlock",
                "NonLinearReadoutBlock",
            ]:
                assert hasattr(readout, "linear_1") or hasattr(
                    readout, "linear_mid"
                ), "Readout block must have linear_1 or linear_mid"
                if hasattr(readout, "linear_1"):
                    shape_input_1 = (
                        model_foundations.readouts[i]
                        .linear_1.__dict__["irreps_out"]
                        .num_irreps
                    )
                    shape_output_1 = readout.linear_1.__dict__["irreps_out"].num_irreps
                else:
                    raise ValueError("Readout block must have linear_1")
                if hasattr(readout, "linear_1"):
                    model_readouts_one_linear_1_weight = readout.linear_1.weight.clone()
                    model_readouts_one_linear_1_weight = (
                        model_foundations.readouts[i]
                        .linear_1.weight.view(num_channels_foundation, -1)
                        .repeat(1, len(model_heads))
                        .flatten()
                        .clone()
                    )
                    readout.linear_1.weight = torch.nn.Parameter(
                        model_readouts_one_linear_1_weight
                    )
                    if (
                        readout.linear_1.bias is not None
                        and readout.linear_1.bias.numel() > 0
                    ):
                        model_readouts_one_linear_1_bias = (
                            model_foundations.readouts[i]
                            .linear_1.bias.view(-1)
                            .repeat(len(model_heads))
                            .clone()
                        )
                        readout.linear_1.bias = torch.nn.Parameter(
                            model_readouts_one_linear_1_bias
                        )
                if hasattr(readout, "linear_mid"):
                    readout.linear_mid.weight = torch.nn.Parameter(
                        model_foundations.readouts[i]
                        .linear_mid.weight.view(
                            shape_input_1,
                            shape_input_1,
                        )
                        .repeat(len(model_heads), len(model_heads))
                        .flatten()
                        .clone()
                        / ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    # if it has biases transfer them too
                    if (
                        readout.linear_mid.bias is not None
                        and readout.linear_mid.bias.numel() > 0
                    ):
                        readout.linear_mid.bias = torch.nn.Parameter(
                            model_foundations.readouts[i]
                            .linear_mid.bias.repeat(len(model_heads))
                            .clone()
                        )
                if hasattr(readout, "linear_2"):
                    model_readouts_one_linear_2_weight = readout.linear_2.weight.clone()
                    model_readouts_one_linear_2_weight = model_foundations.readouts[
                        i
                    ].linear_2.weight.view(shape_input_1, -1).repeat(
                        len(model_heads), len(model_heads)
                    ).flatten().clone() / (
                        ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    readout.linear_2.weight = torch.nn.Parameter(
                        model_readouts_one_linear_2_weight
                    )
                    if (
                        readout.linear_2.bias is not None
                        and readout.linear_2.bias.numel() > 0
                    ):
                        model_readouts_one_linear_2_bias = (
                            model_foundations.readouts[i]
                            .linear_2.bias.view(-1)
                            .repeat(len(model_heads))
                            .flatten()
                            .clone()
                        )
                        readout.linear_2.bias = torch.nn.Parameter(
                            model_readouts_one_linear_2_bias
                        )
    _handled_attrs = {"interactions", "products", "readouts"}
    for attr_name, module in model.named_children():
        if attr_name in _handled_attrs:
            continue
        # MACE-Field adds its field-coupling modules to the target model, but
        # a plain foundation checkpoint (including MACE-MH-1) has no matching
        # children.  Leave those newly initialized modules untouched.
        # The checkpoint module registry is the authoritative way to match
        # optional extension modules such as MACEField's field coupling.
        # pylint: disable-next=protected-access
        if attr_name not in model_foundations._modules:
            continue
        submodules = (
            list(zip(module, model_foundations.__dict__["_modules"][attr_name]))
            if isinstance(module, torch.nn.ModuleList)
            else [(module, getattr(model_foundations, attr_name))]
        )
        for sub_new, sub_found in submodules:
            for emb_name in ("source_embedding", "target_embedding"):
                if not hasattr(sub_new, emb_name):
                    continue
                emb_new = getattr(sub_new, emb_name)
                emb_found = getattr(sub_found, emb_name)
                if (
                    hasattr(emb_new, "weight")
                    and hasattr(emb_found, "weight")
                    and emb_found.weight.shape[0]
                    == num_species_foundations * num_channels_foundation
                    and emb_new.weight.shape[0] == num_species * num_channels_foundation
                ):
                    emb_new.weight = torch.nn.Parameter(
                        emb_found.weight.view(num_species_foundations, -1)[
                            indices_weights, :
                        ]
                        .flatten()
                        .clone()
                        / (num_species_foundations / num_species) ** 0.5
                    )

    if getattr(model_foundations, "scale_shift", None) is not None and hasattr(
        model, "scale_shift"
    ):
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift and not field_target:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()
        elif use_shift and field_target and "pt_head" in model_heads:
            # Replay labels are generated by the selected single-head
            # foundation model.  Its structural readout shift must therefore
            # be restored for pt_head, while the real field heads retain
            # their dataset-specific residual shifts.
            pt_head_index = model_heads.index("pt_head")
            foundation_shift = model_foundations.scale_shift.shift.reshape(-1)[0].clone()
            if model.scale_shift.shift.numel() == 1:
                model.scale_shift.shift = foundation_shift
            else:
                model.scale_shift.shift[pt_head_index] = foundation_shift

    if field_target and load_readout and is_mace_foundation_model(model_foundations):
        _initialize_field_modules(model, scale=1.0e-3)

    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if field_target and (
            name == "scale_shift.shift"
            or name.startswith("atomic_energies_fn.")
        ):
            continue
        if not load_readout and name.startswith("readouts."):
            continue
        if not load_readout and name.startswith(("field_feats.", "field_linear.")):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)

    model.to(target_dtype)

    return model


def _initialize_field_modules(
    model: torch.nn.Module, scale: float = 1.0e-3
) -> None:
    """Keep newly added field-response modules quiet but trainable.

    A plain MACE foundation has no parameters corresponding to the legacy
    field coupling. Random full-size initialization can therefore inject a
    very large field-dependent energy into the first fine-tuning step. Scaling
    the target-only weights leaves them nonzero, so they receive gradients and
    can move immediately, while the transferred structural backbone remains
    the initial predictor.
    """
    if scale <= 0.0:
        raise ValueError("field response initialization scale must be positive")
    module_names = []
    if hasattr(model, "field_feats"):
        module_names.extend(("field_feats", "field_linear"))
    with torch.no_grad():
        for module_name in module_names:
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.mul_(scale)


def load_foundations_elements_magnetic(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)

    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_mag_radial = model.mag_radial_embedding.num_basis
    num_species = len(indices_weights)
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    _copy_radial_weights(model, model_foundations)

    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[: num_radial + num_mag_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )

        # conv_tp_weights_magmom
        for j in range(1):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(
                    model.interactions[i].conv_tp_weights_magmom, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights_magmom,
                        layer_name,
                    ).weight.clone()
                )
            else:
                getattr(
                    model.interactions[i].conv_tp_weights_magmom, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights_magmom,
                        layer_name,
                    ).weight.clone()
                )

        model.interactions[i].magmom_linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].magmom_linear.weight.clone()
        )
        if model.interactions[i].__class__.__name__ in [
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
        ]:
            model.interactions[i].magmom_skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .magmom_skip_tp.weight.flatten()
                .clone()
            )
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight.flatten().clone()
            )
        if model.interactions[i].__class__.__name__ in [
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
            "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
        ]:
            # Assuming only 1 layer in density_fn
            getattr(model.interactions[i].density_fn, "layer0").weight = (
                torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].density_fn,
                        "layer0",
                    ).weight.clone()
                )
            )
    # Transferring products
    for i in range(2):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            for k in range(2):  # Assuming 2 weights in each contraction
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        model.products[i].conv_tp.weight = torch.nn.Parameter(
            model_foundations.products[i].conv_tp.weight.clone()
        )
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.products[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.products[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[:num_mag_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.products[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.products[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )
        model.products[i].linear_ori.weight = torch.nn.Parameter(
            model_foundations.products[i].linear_ori.weight.clone()
        )
        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    if load_readout:
        # Transferring readouts
        model_readouts_zero_linear_weight = model.readouts[0].linear.weight.clone()
        model_readouts_zero_linear_weight = (
            model_foundations.readouts[0]
            .linear.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[0].linear.weight = torch.nn.Parameter(
            model_readouts_zero_linear_weight
        )

        shape_input_1 = (
            model_foundations.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        )
        shape_output_1 = model.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        model_readouts_one_linear_1_weight = model.readouts[1].linear_1.weight.clone()
        model_readouts_one_linear_1_weight = (
            model_foundations.readouts[1]
            .linear_1.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[1].linear_1.weight = torch.nn.Parameter(
            model_readouts_one_linear_1_weight
        )
        model_readouts_one_linear_2_weight = model.readouts[1].linear_2.weight.clone()
        model_readouts_one_linear_2_weight = model_foundations.readouts[
            1
        ].linear_2.weight.view(shape_input_1, -1).repeat(
            len(model_heads), len(model_heads)
        ).flatten().clone() / (
            ((shape_input_1) / (shape_output_1)) ** 0.5
        )
        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_readouts_one_linear_2_weight
        )
    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()
    return model


def load_foundations(
    model,
    model_foundations,
    include_readouts: bool = False,
):
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if not include_readouts and name.startswith("readouts."):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)
    return model


def load_foundations_mdp(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    max_L: int = 2,
):
    """
    Transfer weights from a pretrained AtomicDielectricMACE to a new one,
    with species remapping for a (possibly smaller) element set.

    Unlike load_foundations_elements, this handles higher-order irreps
    in skip_tp and transfers all angular momentum channels in products.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in table.zs]
    num_species = len(indices_weights)
    num_radial = model.radial_embedding.out_dim
    species_scale = (num_species_foundations / num_species) ** 0.5

    # --- Node embedding: extract rows for target species ---
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / species_scale
    )

    # --- Radial embedding ---
    _copy_radial_weights(model, model_foundations)

    # --- Interactions ---
    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])
        if hasattr(model.interactions[i], "linear"):
            model.interactions[i].linear.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_1"):
            model.interactions[i].linear_1.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_1.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_2"):
            model.interactions[i].linear_2.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_2.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_res"):
            model.interactions[i].linear_res.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_res.weight.clone()
            )
        if hasattr(model.interactions[i], "source_embedding"):
            model.interactions[i].source_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .source_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "target_embedding"):
            model.interactions[i].target_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .target_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        # skip_tp: use general reshape [-1, N_sp, N_ch] to handle higher-order irreps
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight
            )
        else:
            foundation_skip = model_foundations.interactions[i].skip_tp.weight
            rest_dim = foundation_skip.numel() // (
                num_species_foundations * num_channels_foundation
            )
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                foundation_skip.reshape(
                    rest_dim, num_species_foundations, num_channels_foundation
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # --- Products: transfer ALL angular momentum channels (not just L=0 for last) ---
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        # MDP readouts use all irreps, so always transfer all contractions
        max_range = max_L + 1
        for j in range(max_range):
            product.symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights_prod, :, :]
                    .clone()
                )
            )
            target_weights = product.symmetric_contractions.contractions[j].weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                target_weights[k] = torch.nn.Parameter(
                    source_weights[k][indices_weights_prod, :, :].clone()
                )
        product.linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    # --- Readouts: copy matching params by name+shape (species-independent) ---
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if not name.startswith("readouts."):
            continue
        if name not in model_state:
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)

    return model
