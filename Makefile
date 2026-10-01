HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)

.PHONY: install ruff

install:
	@if [ "$(HAS_GPU)" -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi


ruff:
	uv run ruff check --fix --unsafe-fixes --extend-select I
	uv run ruff format
