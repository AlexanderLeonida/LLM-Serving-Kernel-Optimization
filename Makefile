.PHONY: install dev test bench profile lint clean figures ablation

install:
	pip install -e .

dev:
	pip install -e ".[dev,serving,profile]"

test:
	pytest tests/ -v --tb=short

test-gpu:
	pytest tests/ -v --tb=short -m gpu

bench:
	pytest benchmarks/ -v --benchmark-only

bench-cli:
	lightning-router bench --kernel all --batch-size 32 --num-experts 4

ablation:
	python benchmarks/ablation_study.py --output-dir results/ablation

figures:
	python benchmarks/generate_figures.py --results-dir results/ --output-dir figures/

profile-kernel:
	lightning-router profile --kernel expert_routing --output-dir profiling_results/

profile-e2e:
	lightning-router profile --kernel moe_layer --output-dir profiling_results/

serve:
	lightning-router serve --config configs/moe_4expert.yaml --port 8000

lint:
	ruff check lightning_router/ tests/ benchmarks/
	mypy lightning_router/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
