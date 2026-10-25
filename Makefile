.PHONY: install generate bench inspect bootstrap-source repro v1 v2 diff dag diversity contamination sample tokenize check check-hw1 check-hw2 check-hw3 check-hw4 check-hw5 check-unit check-all train train-all train-freeze compare plot clean clean-training distclean

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
	@set -e; check_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/mlops-hw5-check.XXXXXX"); trap 'rm -rf "$$check_tmp"' EXIT INT TERM; TMPDIR="$$check_tmp" bash tests/check.sh
	uv run python scripts/check_hw5_artifacts.py

check-hw1:
	bash tests/check_hw1.sh

check-hw2:
	bash tests/check_hw2.sh

check-hw3:
	bash tests/check_hw3.sh

check-hw4:
	bash tests/check_hw4.sh

check-hw5: check

check-unit:
	uv run python -m unittest discover -s tests -p 'test_*.py'

check-all:
	$(MAKE) check-unit
	$(MAKE) check-hw1
	$(MAKE) check-hw2
	$(MAKE) check-hw3
	$(MAKE) check-hw4
	$(MAKE) check
	$(MAKE) check-hw67
	$(MAKE) check-eval

clean: clean-training
	rm -f docs/bench.json docs/report.json out1.txt out2.txt params.yaml.bak
	rm -rf src/__pycache__ scripts/__pycache__ tests/__pycache__

# Два обучения идут последовательно, чтобы не удваивать память.
train:
	$(MAKE) train-all
	$(MAKE) train-freeze
	$(MAKE) plot

train-all:
	uv run python -m src.train --variant all_layers

train-freeze:
	uv run python -m src.train --variant freeze14

compare:
	uv run python -m src.compare --variant all_layers

plot:
	uv run python -m src.plot

clean-training:
	rm -rf models
	rm -f metrics/train_*.json metrics/compare_*.json docs/curves.png docs/compare.md .check_*.log

distclean: clean-training
	rm -rf .venv

# ДЗ 6+7 продолжает общий проект; conf/hw67.yaml не меняет параметры ДЗ1–5.
.PHONY: up down sweep report candidate baseline-init evaluate eval check-eval check-negative promote check-storage check-hw67 baseline-update predict predict-candidate
up:
	docker compose -p mlops-hw6 up --build -d --wait

down:
	docker compose -p mlops-hw6 down

sweep:
	uv run python -m src.sweep

report:
	uv run python -m src.experiment_report

candidate:
	uv run python -m evaluation.cli candidate

baseline-init:
	uv run python -m evaluation.cli baseline-init

evaluate:
	uv run python -m evaluation.cli evaluate

eval:
	uv run python -m evaluation.cli gate

check-eval:
	uv run python -m unittest discover -s tests -p 'test_eval*.py' -v
	uv run python -m unittest discover -s tests -p 'test_pipeline*.py' -v

check-negative:
	uv run python -m evaluation.cli negative

promote:
	uv run python -m evaluation.cli promote

check-storage:
	uv run python tests/check_storage.py

check-hw67:
	bash tests/check_hw67.sh

baseline-update:
	uv run python -m evaluation.cli baseline-update

predict:
	uv run python -m src.predict --uri models:/my-qwen-adapter@champion --prompt "Выберите жанр фильма: история о путешествии космонавтов."

predict-candidate:
	uv run python -m src.predict --uri models:/my-qwen-adapter@candidate --prompt "Выберите жанр фильма: история о путешествии космонавтов."

.PHONY: check-hw67-all
check-hw67-all: check-hw67 check-eval
