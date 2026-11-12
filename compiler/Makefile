PYTHON ?= python3

.PHONY: help test binary
help:
	@echo "Compile: $(PYTHON) main.py model.onnx [--calibration samples.npy] [--debug]"
	@echo "Test:    make test"
	@echo "Package: make binary (requires PyInstaller; see README.md)"

test:
	$(PYTHON) -m unittest test_compiler test_memory -v

binary:
	$(PYTHON) build_binary.py
