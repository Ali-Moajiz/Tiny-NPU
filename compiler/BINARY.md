# Standalone compiler executable — v0.3.0

Executable: **`Tiny-NPU/compiler/bin/tiny-npu-compiler`**. It bundles Python,
NumPy, ONNX and compiler source into a Linux x86-64 ELF executable. It needs no
Python installation, pip environment or network at runtime. This ZIP replaces
only the `compiler/` directory; nothing in `c-model/` is included or changed.

## Compile with one command

From your `Tiny-NPU/` project root:

```sh
chmod +x compiler/bin/tiny-npu-compiler
./compiler/bin/tiny-npu-compiler model_qdq.onnx
./compiler/bin/tiny-npu-compiler model.onnx --calibration samples.npy
```

The first command uses embedded quantization in compatible PTQ/QAT Q/DQ exports.
The second performs PTQ on a float model. Both default to batch 4 and write to
`MODEL_STEM_compiled/` beside the model. Override with `--batch 1` or `--out DIR`.
Float models search `model.calibration.npy`, then `calibration.npy` beside the
model if no explicit calibration path is supplied. Missing calibration is an
error; synthetic samples are never substituted. Quantized models reject an
explicit calibration argument and ignore sidecar calibration files.

```sh
./compiler/bin/tiny-npu-compiler compiler/examples/qat_qdq.onnx --out compiled
./compiler/bin/tiny-npu-compiler disasm compiled/program.bin
./compiler/bin/tiny-npu-compiler run --image compiled --inputs compiler/examples/inputs.npy --out compiled/run.json
./compiler/bin/tiny-npu-compiler --version
```

The executable can be copied elsewhere or renamed to `compiler-binary`.
`compiler-binary model.onnx` then works without the Python source directory.
`compile --model ...` remains supported. Models using ONNX external tensor files
must retain those files at their model-relative locations.

Only the documented dense/ReLU subset is supported. Q/DQ must use signed INT8,
zero point 0, scalar per-tensor scales. See `docs/QUANTIZED_IMPORT.md`.

## Platform

Built with CPython 3.11.18 and versions pinned in `requirements-build.txt`.
Runtime-tested on Linux x86-64 with glibc 2.39. See
`docs/binary_abi_audit.json` for ELF symbol requirements and
`docs/VALIDATION.md` for actual checks. Audited maxima are GLIBC 2.17 and
external GLIBCXX 3.4.18. ABI inspection is not a runtime test
on every older distribution. System libstdc++ and libgcc are required.

Not a Windows, ARM64 or Alpine/musl binary. PyInstaller extracts its embedded
libraries into temporary storage; that storage must allow execution. Set
`TMPDIR` to a writable, executable filesystem if the default is mounted noexec.
Third-party license texts are included in `third_party_licenses/`.

## Rebuild (inside compiler/)

```sh
python3 -m venv .venv-build
. .venv-build/bin/activate
python -m pip install -r requirements-build.txt
make binary PYTHON=python
```

Output is `bin/tiny-npu-compiler`; temporary build products are under `build/`.
Build using a Python and system libraries compatible with the oldest Linux you
intend to support. A newer system Python may raise the required glibc version.
The supplied build uses uv's python-build-standalone distribution. Byte-for-byte
reproducibility across different build environments is not claimed.

This is a compiler and functional ISA interpreter. It neither updates nor
validates the C model's programmable decoder. ISA v1 remains unchanged.
