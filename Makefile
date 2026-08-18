.PHONY: install format lint test coverage fonts train benchmark preview api demo docker clean

install:
	python -m pip install -e '.[dev,demo]'
	pre-commit install || true

format:
	python -m ruff format src tests scripts
	python -m ruff check --fix src tests scripts

lint:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts
	python -m mypy src/fontprint

test:
	python -m pytest

coverage:
	python -m pytest --cov=fontprint --cov-report=term-missing --cov-report=html --cov-fail-under=85

fonts:
	python scripts/fetch_ofl_fonts.py --destination data/fonts

train:
	fontprint train --config configs/base.yaml --font-root data/fonts

benchmark:
	fontprint benchmark --checkpoint artifacts/fontprint.pt --font-root data/fonts --documents 40 --markdown

preview:
	fontprint synthesize --output docs/example-case.png --tampered

api:
	fontprint serve --checkpoint artifacts/fontprint.pt

demo:
	fontprint demo --checkpoint artifacts/fontprint.pt

docker:
	docker build -t fontprint:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
