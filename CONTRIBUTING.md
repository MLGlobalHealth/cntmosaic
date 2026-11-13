# Contributing to cntmosaic

Thank you for your interest in `cntmosaic`. This is a preliminary (v0.6) release shared with
collaborators and early users. Contributions in the form of bug reports and feature requests
are very welcome.

## Reporting a Bug

If you encounter unexpected behaviour, please [open a bug report](https://github.com/MLGlobalHealth/cntmosaic/issues/new?template=bug_report.yml).

Before submitting, please:
- Check the [existing issues](https://github.com/MLGlobalHealth/cntmosaic/issues) to see if the bug has already been reported.
- Include a minimal reproducible example if possible.
- Include the output of `python -c "import cntmosaic; print(cntmosaic.__version__)"` and your Python version.

## Requesting a Feature

If you have an idea for a new feature or an improvement, please [open a feature request](https://github.com/MLGlobalHealth/cntmosaic/issues/new?template=feature_request.yml).

Please describe:
- The problem you are trying to solve or the use case you have in mind.
- How you would expect the feature to behave.

## Questions

For questions about usage or methodology, please open a [GitHub Discussion](https://github.com/MLGlobalHealth/cntmosaic/discussions) rather than an issue.

## Code Contributions

Code contributions via pull requests are welcome, but please open an issue first to discuss
the proposed change. This avoids wasted effort if the change is out of scope for the current
development priorities.

When submitting a pull request:
1. Fork the repository and create a branch from `main`.
2. Install the package in editable mode with dev dependencies: `pip install -e ".[dev]"`.
3. Add or update tests for any changed behaviour.
4. Run the test suite before submitting: `pytest` from the repository root. For a coverage report, use
   `pytest --cov=cntmosaic --cov-report=html`; to run the tests of one module, pass its folder, e.g.
   `pytest cntmosaic/models/tests/`.
5. Follow existing code style (enforced by `black` and `flake8`).

### Tutorial notebooks

The tutorials are Jupyter notebooks in [`docs/source/tutorials/`](docs/source/tutorials/). Their outputs are stripped by [`nbstripout`](https://github.com/kynan/nbstripout) when they are committed, which keeps the repository small. Set it up once after cloning:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

The documentation build executes the notebooks to produce their outputs (see `docs/source/conf.py`). To build the documentation locally:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

## Licence

By contributing, you agree that your contributions will be licensed under the
[BSD 3-Clause Licence](LICENSE) that covers this project.
