.PHONY: install generate bench inspect bootstrap-source repro v1 v2 diff dag diversity contamination sample tokenize check check-hw1 check-hw2 check-hw3 check-all clean

install:
	uv sync

generate:
	uv run python -m src.generate

bench:
	uv run python -m src.bench

inspect:
	uv run python -m src.inspect_model

bootstrap-source:
	uv run python scripts/fetch_source.py
	uv run dvc commit sources/movies-dataset.dvc

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

sample:
	uv run python -m scripts.make_sample

tokenize:
	uv run python -m src.tokenize_data

check:
	bash tests/check.sh

check-hw1:
	bash tests/check_hw1.sh

check-hw2:
	bash tests/check_hw2.sh

check-hw3:
	bash tests/check_hw3.sh

check-all:
	$(MAKE) check-hw1
	$(MAKE) check-hw2
	$(MAKE) check-hw3
	$(MAKE) check

clean:
	rm -f docs/bench.json docs/report.json out1.txt out2.txt params.yaml.bak
	rm -rf src/__pycache__ scripts/__pycache__ tests/__pycache__
