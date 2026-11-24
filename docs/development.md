# Development

## Development setup

Install the development dependencies and repository hooks from the root of the
source checkout:

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
