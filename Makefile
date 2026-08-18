.PHONY: install test lint fonts train preview api demo docker

install:
	python -m pip install -e '.[dev,demo]'

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts
	python -m mypy src/fontprint

fonts:
	python scripts/fetch_ofl_fonts.py --destination data/fonts

train:
	fontprint train --config configs/base.yaml --font-root data/fonts

preview:
	fontprint synthesize --output docs/example-case.png --tampered

api:
	fontprint serve --checkpoint artifacts/fontprint.pt

demo:
	fontprint demo --checkpoint artifacts/fontprint.pt

docker:
	docker build -t fontprint:latest .
