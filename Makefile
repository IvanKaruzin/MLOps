.PHONY: install generate bench inspect repro v1 v2 diff dag diversity contamination check check-hw1 check-hw2 check-all clean

install:
	uv sync

generate:
	uv run python -m src.generate

bench:
	uv run python -m src.bench

inspect:
	uv run python -m src.inspect_model

repro:
	uv run dvc repro

v1:
	uv run python scripts/set_version.py v1
	uv run dvc repro

v2:
	uv run python scripts/set_version.py v2
	uv run dvc repro

diff:
	uv run dvc metrics diff

dag:
	uv run dvc dag

diversity:
	uv run python -m src.diversity

contamination:
	uv run python scripts/check_contamination.py

check:
	bash tests/check.sh

check-hw1:
	bash tests/check_hw1.sh

check-hw2:
	bash tests/check_hw2.sh

check-all:
	$(MAKE) check-hw1
	$(MAKE) check-hw2
	$(MAKE) check

clean:
	rm -f docs/bench.json docs/report.json out1.txt out2.txt params.yaml.bak
	rm -rf src/__pycache__ scripts/__pycache__ tests/__pycache__
