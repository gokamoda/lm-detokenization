HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_GPU),1)
TORCH_EXTRA := cu121
else
TORCH_EXTRA := default
endif

.PHONY: install

install:
	@echo "=== Install with torch extra: $(TORCH_EXTRA) ==="
	uv sync --extra=$(TORCH_EXTRA) --no-dev
	uv sync --extra=$(TORCH_EXTRA) --dev --no-build-isolation
