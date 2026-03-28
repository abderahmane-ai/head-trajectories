.PHONY: help install test lint format clean docs docker-build docker-shell docker-notebook

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove generated files"
	@echo "  make docs       - Generate documentation"
	@echo "  make docker-build    - Build the Docker image"
	@echo "  make docker-shell    - Start an interactive shell in the container"
	@echo "  make docker-notebook - Launch JupyterLab in the container"

install:
	pip install -r requirements.txt

test:
	python run_tests.py

test-cov:
	pytest tests/ --cov=probing --cov=model --cov=analysis --cov-report=html --cov-report=term

lint:
	flake8 probing/ model/ analysis/ training/ data/ visualization/
	mypy probing/ model/ analysis/ --ignore-missing-imports

format:
	black probing/ model/ analysis/ training/ data/ visualization/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf build dist *.egg-info

docs:
	@echo "Documentation available in docs/"
	@echo "  - QUICKSTART.md: Getting started guide"
	@echo "  - ARCHITECTURE.md: Codebase overview"
	@echo "  - FAQ.md: Frequently asked questions"

docker-build:
	docker build -t head-trajectories:latest .

docker-shell:
	docker run --rm -it --gpus all -v "$$(pwd):/workspace/head-trajectories" -w /workspace/head-trajectories head-trajectories:latest

docker-notebook:
	docker run --rm -it --gpus all -p 8888:8888 -v "$$(pwd):/workspace/head-trajectories" -w /workspace/head-trajectories head-trajectories:latest jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root --ServerApp.token=head-trajectories
