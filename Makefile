HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)

.PHONY: install install-cu118 ruff

install:
	@if [ "$(HAS_GPU)" -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

# For a server whose NVIDIA driver is too old for the CUDA 12 build of PyTorch
# (e.g. a driver of CUDA 11.4): the CUDA 11.8 build, which runs on any CUDA 11
# driver from 450.80.02.
install-cu118:
	uv sync --all-groups --extra=gpu-cu118


ruff:
	uv run ruff check --fix --unsafe-fixes --extend-select I
	uv run ruff format
