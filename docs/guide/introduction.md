# Introduction

MACE-Field extends `ScaleShiftMACE` with a uniform electric-field input. The
field is represented as an equivariant vector and coupled to latent features
between interaction layers. The MACE energy readout produces one scalar
field-dependent energy for each configuration.

## Energy and response properties

For energy $E(\mathbf{R}, \boldsymbol{\mathcal{E}})$, MACE-Field derives the
response properties from derivatives of that same scalar:

- Polarization is obtained from the field derivative,
  $\mathbf{P} = -\Omega^{-1}\,\partial E/\partial\boldsymbol{\mathcal{E}}$,
  where $\Omega$ is the cell volume for periodic configurations.
- Born effective charges are the volume-scaled position derivative of
  polarization, $Z^*_{\kappa,\alpha\beta}=\Omega\,\partial P_\alpha/\partial R_{\kappa,\beta}$.
- The `polarizability` output is the relative electric susceptibility,
  $\chi_{\alpha\beta}=\epsilon_0^{-1}\partial P_\alpha/\partial\mathcal{E}_\beta$.
  It is dimensionless and is reported in $\epsilon_0$ units.

This shared energy makes the energy, forces, and field responses consistent
with one another. At zero field, the field coupling contributes zero, which
allows the MACE backbone to be initialized from a compatible foundation model.

## Units and conventions

The uniform field is given in V/Å. Reference keys use the `REF_` prefix by
default; field response keys are `REF_electric_field`,
`REF_polarization`, `REF_becs`, and `REF_polarizability`. See [Data](data.md)
for shapes and units.

The polarization sign convention is
$\mathbf{P}=-\partial E/\partial\boldsymbol{\mathcal{E}}$. For periodic materials, polarization is
multivalued by a lattice quantum; use reference data and the periodic branch
handling in `universal_field` consistently.
