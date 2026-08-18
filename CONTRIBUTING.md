# Contributing

Contributions that improve calibration, capture-domain robustness, region grouping, or evaluation on openly licensed datasets are welcome.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,demo]'
pre-commit install
pytest
ruff check src tests
```

Keep generated data, proprietary fonts, real documents, and checkpoints out of commits. Add tests for behavioral changes and update the model/data cards when a change affects intended use, evaluation, or risk. Pull requests should describe the font collection and seed behind any reported metrics.

By participating, you agree to keep discussion constructive and focused on the work. Security or privacy issues should follow the private process in `SECURITY.md` rather than a public issue.
