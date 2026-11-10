# Compiler v0.3.0 validation

## Source tests

Environment: CPython 3.11.18, NumPy 1.26.4, ONNX 1.17.0,
ONNX Runtime 1.20.1. All **32 unittest tests passed**; no skipped tests in
this run. See `test_results.txt`. Install `requirements-test.txt` to reproduce
including the optional real-export test; ordinary requirements omit ONNX Runtime.

- Existing encoding, independent integer oracle, every per-k accumulator,
  output tile, final result, overflow/capacity, loops and CLI tests still pass.
- Original multi-tile matrix: 7 shapes × 4 batches × 2 forwarding settings =
  56 builds, each with 9 real vectors and padded final batches.
- Q/DQ source comparison: 3 shapes × 2 weight representations × 2 I/O dtypes ×
  2 dense forms × 4 batches × 2 forwarding settings = 192 compiled executions.
  Power-of-two scale fixtures match ONNX ReferenceEvaluator outputs exactly.
- Additional checks cover independent ReLU quantization boundaries, fractional
  QAT-style weight rounding, float bias error, external tensor data, opsets
  13/17/19/21, missing boundaries, unsupported schemes and invalid scales.
- A real ONNX Runtime symmetric per-tensor PTQ export compiled without
  calibration. 33 input vectors passed the per-k integer oracle checks and
  differed by at most one INT8 code from source ONNX Runtime Q/DQ execution
  with graph optimization disabled. This bounded comparison is specific to
  that test, not a universal quantized-model accuracy guarantee.
- PyTorch/TensorFlow QAT training/export was not executed. The QAT fixture tests
  the standard FLOAT -> Q -> DQ weight representation, not every framework's
  export dialect. PTQ/QAT provenance itself is not inferred from the graph.

## Regression and executable

The original float example's `program.bin` and `sram.bin` are byte-identical
before and after this update. ISA v1 and SRAM address/packing contracts did
not change. Compiled examples include fresh manifests/import reports/run traces.

`binary_smoke.json` records standalone-executable checks in an isolated working
directory with no compiler source and PATH containing no Python. Checks include
model-only compilation for both quantized weight representations and INT8 I/O,
raw input execution with a partial final batch, disassembly, float calibration
failure and sidecar success, real exporter input at batch 1 without forwarding,
and unsupported-zero-point rejection. Frozen and source example bytes match.

The ELF bootloader plus 20 bundled libraries/extensions were inspected.
Maximum required GLIBC: **2.17**; external libstdc++ GLIBCXX: **3.4.18**;
CXXABI: **1.3.5**. See `binary_abi_audit.json`. Runtime execution was tested on
Linux x86-64/glibc 2.39. Older distribution compatibility is based on ABI
inspection, not a VM test. The binary requires system libstdc++/libgcc and an
executable temporary directory.

## Scope

This release contains only `compiler/`, including its executable in `bin/`.
No C-model code or architecture drawings were changed. The interpreter is
functional, not cycle-accurate. No RTL throughput or model task-accuracy claim
is made. The C model needs the documented programmable ISA implementation to
execute this image; compiler-only tests cannot establish that integration.
