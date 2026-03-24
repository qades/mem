# Context Management Benchmark Makefile

.PHONY: all venv install test lint run-benchmark datasets datasets-convert datasets-all clean help

PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := source $(VENV_DIR)/bin/activate

all: install

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

# Install dependencies
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed"

# Run tests
test: install
	$(VENV_PYTHON) test_benchmark.py

# Run linter (ruff)
lint: install
	$(PIP) install ruff
	$(VENV_PYTHON) -m ruff check .

# Run benchmarks
run-benchmark: install
	$(VENV_PYTHON) run_benchmark.py --compare

# Run specific benchmark config
run-config: install
	$(VENV_PYTHON) run_benchmark.py --config $(config)

# Download and process datasets
datasets: install
	$(VENV_PYTHON) download_datasets.py --output-dir $(data_dir) --datasets $(dataset) $(dataset_args)

# Download and convert datasets to unified format
datasets-convert: install
	$(VENV_PYTHON) download_datasets.py --output-dir $(data_dir) --datasets $(dataset) --convert $(dataset_args)

# Download all datasets
datasets-all: install
	$(VENV_PYTHON) download_datasets.py --output-dir ./data --datasets all --convert

# Download specific dataset subset
dataset-babilong: install
	$(VENV_PYTHON) download_datasets.py --output-dir ./data --datasets babilong --subset $(subset) $(dataset_args)

dataset-mutual: install
	$(VENV_PYTHON) download_datasets.py --output-dir ./data --datasets mutual $(dataset_args)

dataset-agentbench: install
	$(VENV_PYTHON) download_datasets.py --output-dir ./data --datasets agentbench $(dataset_args)

dataset-prolong: install
	$(VENV_PYTHON) download_datasets.py --output-dir ./data --datasets prolong --version $(version) $(dataset_args)

# Clean up
clean:
	rm -rf $(VENV_DIR)
	rm -rf benchmark_results
	rm -rf data
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	@echo "Cleaned up build artifacts"

# Help
help:
	@echo "Context Management Benchmark"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Install dependencies in virtual environment"
	@echo "  make test             Run unit tests"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make run-benchmark    Run comparison of all strategies"
	@echo "  make run-config       Run specific benchmark config"
	@echo ""
	@echo "Dataset commands:"
	@echo "  make datasets-all     Download and convert all datasets"
	@echo "  make datasets         Download datasets (use dataset=, data_dir=)"
	@echo "  make dataset-babilong Download BABILong dataset"
	@echo "  make dataset-mutual   Download MuTual dataset"
	@echo "  make dataset-agentbench Download AgentBench dataset"
	@echo "  make dataset-prolong  Download ProLong dataset"
	@echo ""
	@echo "  make clean            Clean up build artifacts"
	@echo "  make help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make test"
	@echo "  make run-benchmark"
	@echo "  make run-config config=config/knowledge_graph.json"
	@echo "  make datasets-all"
	@echo "  make datasets dataset=babilong subset=128k data_dir=./data"
	@echo "  make dataset-babilong subset=1m"
