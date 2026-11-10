#!/usr/bin/env python3
"""Project-local compiler entry point; usable from any working directory.

Example, from the Tiny NPU root:
    python3 compiler/npucc.py compile --model model.onnx \
        --calibration calibration.npy --out build/compiled_model

Python makes this script's directory importable. No package install or PYTHONPATH
setting is required. The input/output paths remain relative to the caller.
"""
from tiny_npu.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
