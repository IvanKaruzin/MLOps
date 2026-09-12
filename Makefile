.PHONY: install generate bench inspect check check-hw1 check-all clean

install:
	uv sync

generate:
	uv run python -m src.generate

bench:
	uv run python -m src.bench

inspect:
	uv run python -m src.inspect_model

check:
	bash tests/check.sh

check-hw1:
	bash tests/check_hw1.sh

check-all:
	$(MAKE) check-hw1
	$(MAKE) check

clean:
	rm -f docs/bench.json docs/report.json out1.txt out2.txt params.yaml.bak
	rm -rf src/__pycache__ tests/__pycache__
