MACE-Field documentation
========================

MACE-Field extends MACE with a uniform electric-field input. It predicts one
field-dependent scalar energy and obtains polarization, Born effective charges,
and polarizability by differentiating that energy.

This guide covers MACE-Field-specific workflows. For general MACE installation,
training options, acceleration backends, and foundation-model details, see the
`MACE documentation <https://mace-docs.readthedocs.io/>`_.

Getting started
---------------

Start with the :doc:`quick_start` to install this repository and evaluate the
released MH-0 checkpoint.

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quick_start
   user_guide/data
   user_guide/training
   user_guide/inference
   user_guide/hdf5

Model outputs
-------------

For a model energy :math:`E(\mathbf{R}, \boldsymbol{\mathcal{E}})`, the field
response is derived from the same scalar:

.. math::

   \mathbf{P} = -\frac{1}{\Omega}
   \frac{\partial E}{\partial \boldsymbol{\mathcal{E}}},
   \qquad
   Z^{*}_{\kappa,\alpha\beta} =
   \frac{\partial P_{\alpha}}{\partial R_{\kappa,\beta}},
   \qquad
   \chi_{\alpha\beta} =
   \frac{\partial P_{\alpha}}{\partial \mathcal{E}_{\beta}}.

The volume factor :math:`\Omega` applies to periodic polarization. MACE-Field
uses the repository's documented output units and tensor conventions; see
:doc:`user_guide/data` before preparing reference labels.

Source and releases
-------------------

* `MACE-Field source on GitHub <https://github.com/mdi-group/mace-field>`_
* `Released checkpoints and scripts
  <https://github.com/mdi-group/mace-field/releases>`_
* `MACE documentation <https://mace-docs.readthedocs.io/>`_


Package documentation
---------------------

The package guide covers the current installation, training, data, and
calculator interfaces. The examples section links to the Colab-ready tutorial
notebooks.

.. toctree::
   :maxdepth: 2
   :caption: Package Guide

   package-overview
   quick-start
   guide/index
   examples/index
   references
   development
