.PHONY: install dev test bench profile lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,profile]"

test:
	pytest tests/ -v --tb=short

test-gpu:
	pytest tests/ -v --tb=short -m gpu

bench:
	pytest benchmarks/ -v --benchmark-only

profile-kernel:
	python -m lightning_router.profiling.nsight_runner \
		--kernel expert_routing \
		--batch-size 32 \
		--num-experts 4

profile-e2e:
	python -m lightning_router.profiling.nsight_runner \
		--mode end-to-end \
		--batch-size 32 \
		--num-experts 4

serve:
	python -m lightning_router.serving.server \
		--model-config configs/moe_4expert.yaml \
		--tensor-parallel-size 1

lint:
	ruff check lightning_router/ tests/
	mypy lightning_router/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
