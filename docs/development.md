# Development

## Publishing to PyPI

The `mace-field-torch` distribution is built from this repository and keeps
the Python import name `mace`. Do not install it alongside `mace-torch` in one
environment. PyPI's pending trusted publisher must be configured for this
repository before the first release:

- Project name: `mace-field-torch`
- Owner: `mdi-group`
- Repository: `mace-field`
- Workflow filename: `pypi-release.yaml` (stored at `.github/workflows/pypi-release.yaml`)
- GitHub environment: `pypi`

The workflow builds the source and wheel distributions, checks their metadata,
verifies that the release tag matches `mace.__version__`, and smoke-tests the
built wheel before publishing. The next MACE-Field release is version `1.1.0`
and uses the tag `1.1.0`; a manual run builds and smoke-tests
without publishing. Set up the PyPI pending publisher in the account's
Publishing settings before creating the first PyPI release tag.

Install the development dependencies and repository hooks from the root of the
checkout:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

The MACE-Field extension tests live in `tests/extensions/macefield`. Run them
with:

```bash
python -m pytest tests/extensions/macefield
```

The broader suite is arranged by capability. Unit tests are in `tests/unit`,
CLI workflows are in `tests/workflows`, network-enabled foundation tests are
in `tests/foundations`, and external runtime integrations are in
`tests/integrations`.

## Build the documentation

Install the documentation dependencies and build the HTML site:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. Read the Docs uses
`.readthedocs.yaml` and the same Sphinx configuration.
