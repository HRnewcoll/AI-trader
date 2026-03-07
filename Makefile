.PHONY: test lint format docker docker-down clean report hyperopt train backtest help

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ─── Testing ──────────────────────────────────────────────────────────────────
test:  ## Run all 94 offline tests
	$(PYTHON) -m pytest tests/test_core.py -v

test-fast:  ## Run tests without verbose output
	$(PYTHON) -m pytest tests/test_core.py -q

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:  ## Run flake8 linter
	$(PYTHON) -m flake8 . --max-line-length 100 --ignore E203,W503,E501 \
		--exclude .git,__pycache__,artifacts,reports,logs,rust_core,.venv

format:  ## Auto-format with black + isort
	$(PYTHON) -m black . --line-length 100 --exclude "rust_core|artifacts|reports|logs"
	$(PYTHON) -m isort . --profile black

format-check:  ## Check formatting without changing files
	$(PYTHON) -m black . --check --line-length 100 --exclude "rust_core|artifacts|reports|logs"
	$(PYTHON) -m isort . --check-only --profile black

# ─── Trading Modes ────────────────────────────────────────────────────────────
run:  ## Start paper trading + dashboard (default)
	$(PYTHON) run.py

paper:  ## Explicit paper-trading mode
	$(PYTHON) run.py --mode paper

live:  ## Live trading (requires MT5/OANDA credentials in .env)
	$(PYTHON) run.py --mode live

dashboard:  ## Launch Streamlit dashboard only
	$(PYTHON) run.py --dashboard-only

# ─── Model Operations ─────────────────────────────────────────────────────────
train:  ## Train all models (XGBoost + Transformer + RL)
	$(PYTHON) run.py --train-only

retrain:  ## Force retrain even if artifacts exist
	$(PYTHON) run.py --force-retrain --train-only

backtest:  ## Run walk-forward backtest
	$(PYTHON) run.py --backtest

hyperopt:  ## Run Optuna hyperparameter optimisation
	$(PYTHON) run.py --hyperopt

# ─── Reporting ────────────────────────────────────────────────────────────────
report:  ## Generate HTML + JSON QuantStats performance report
	$(PYTHON) run.py --generate-report

# ─── Docker ───────────────────────────────────────────────────────────────────
docker:  ## Start full stack (trader + Prometheus + Grafana)
	docker-compose up --build -d
	@echo "Dashboard → http://localhost:8501"
	@echo "Grafana   → http://localhost:3000  (admin / forex_admin)"
	@echo "Prometheus→ http://localhost:9090"

docker-down:  ## Stop all Docker services
	docker-compose down

docker-logs:  ## Tail Docker logs
	docker-compose logs -f

# ─── Setup ────────────────────────────────────────────────────────────────────
install:  ## Install all Python dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install dev extras (black, isort, flake8, pytest)
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 pytest

setup:  ## First-time setup: install deps + copy .env.example
	$(MAKE) install-dev
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — edit it to add credentials"; fi

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:  ## Remove generated files, caches, and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf artifacts/ reports/ logs/ .mypy_cache/ dist/ build/ *.egg-info/ 2>/dev/null || true
	@echo "Cleaned."

clean-models:  ## Remove only trained model artifacts (keeps logs/reports)
	rm -rf artifacts/models artifacts/onnx artifacts/chromadb 2>/dev/null || true
	@echo "Model artifacts removed."
