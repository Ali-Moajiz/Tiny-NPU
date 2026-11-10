# Quantized ONNX import contract

## Two preparation methods, one hardware target

PTQ calibrates a trained model; QAT trains with simulated quantization. Neither
requires a separate MAC datapath. The export must describe the target's supported
integer arithmetic. This compiler imports **standard ONNX Q/DQ**, not framework
checkpoints, raw `.tflite`, PyTorch fake-quant modules, or arbitrary quantized graphs.
It cannot infer whether training used PTQ or QAT from a file alone.

Accepted representations:

- INT8 weight initializer followed by DequantizeLinear (common PTQ export).
- FLOAT32 weight constant followed by QuantizeLinear and DequantizeLinear
  (a possible QAT export). Constant Q uses nearest-even FLOAT32 arithmetic.
- FLOAT32 input/output with explicit Q/DQ boundaries, or INT8 external I/O
  with DQ at input / Q at output.
- Sequential rank-2 Gemm or MatMul + optional constant Add, optional ReLU.
  Weight transforms Identity/Transpose and Constant(value tensor) are supported.
- Standard-domain ONNX opsets 13..21; external tensor data is loaded relative
  to the model. Each scale is static FLOAT32 scalar or shape [1], finite and
  positive. Zero point is signed INT8 value 0 (INT32 value 0 for bias).
  Scalar and shape-[1] zero-point spellings are both accepted for a scalar scale,
  including the mixed spelling emitted for biases by ONNX Runtime.

Every dense stage must end at a QuantizeLinear boundary. Its activation Q and
DQ parameters must match. No calibration samples are read for this path.
Graph output shapes and feature dimensions are checked; graph branches and
unsupported operators are rejected. The compiler does not rewrite unsupported
schemes into a different quantization scheme.

## Bias and requantization

For sx, sw, sy from the exported model, accumulator units are sx*sw and the
requantizer multiplier represents sx*sw/sy as M / 2**S. Existing INT32 bias
codes are accepted only when their DQ scale matches sx*sw (relative tolerance
2e-7 to allow FLOAT32 scale-product storage). FLOAT32 biases are rounded to
INT32 accumulator units with nearest-even rounding. They are not independently
scaled INT8 biases.

`import_report.json` records source scales, weight representation, bias mode,
maximum absolute bias conversion error, exact scale ratio, M/S ratio and its
absolute error. `quantization.json` contains actual target weights/biases/M/S.
Input, weight and output scales are preserved; finite multiplier precision and
bias conversion still introduce potential error.

**Numerical contract:** emitted binary execution must equal the independent
integer oracle for the lowered parameters. Arbitrary FLOAT32 ONNX Q/DQ execution
is not universally bit-exact to an integer implementation: FLOAT32 intermediate
rounding, bias conversion, input quantization thresholds and finite M/S precision
can change a rounding decision. Validate the exported model's task accuracy on
representative evaluation data. Do not interpret an oracle PASS as an accuracy
measurement against the original trained model.

An already-rounded output followed by its own ReLU Q/DQ boundary cannot simply
be fused across that boundary. It is lowered to an identity-weight dense stage
plus ReLU/requantization. This preserves the intervening clipping and rounding,
but consumes additional weights, instructions and MAC work. Capacity checks still
apply. The import report marks `relu-identity-stage` explicitly.

## Deliberately unsupported

Per-channel/blocked quantization; nonzero zero points; UINT8; dynamic ranges;
QLinearMatMul/QLinearConv and other QOperator graphs; mixed unquantized layers;
convolution; residual branches; activation functions other than ReLU/bypass.
A model using these features needs a compatible export or future compiler AND
hardware support. Supplying a calibration file does not bypass these checks.
The target has only one M/S configuration per layer; per-channel QAT/PTQ is not
silently collapsed to per-tensor scales.

## Commands (from Tiny-NPU/)

```sh
./compiler/bin/tiny-npu-compiler model_qdq.onnx --out compiled
./compiler/bin/tiny-npu-compiler disasm compiled/program.bin
./compiler/bin/tiny-npu-compiler run --image compiled --inputs inputs.npy --out compiled/run.json
# If inputs.npy already holds signed INT8 codes:
./compiler/bin/tiny-npu-compiler run --image compiled --inputs inputs_int8.npy --quantized --out compiled/run.json
```

The runner treats inputs as real float features by default, even when the source
ONNX used INT8 external I/O. Use `--quantized` explicitly for raw codes. Output
JSON contains both raw INT8 and dequantized values using the fixed output scale.

`examples/make_quantized_examples.py` builds tiny deterministic Q/DQ fixtures
in the two weight representations; these are export-format examples, not trained
PTQ/QAT models. The training framework and its exporter remain your choice.

## Reproduce the real-export integration test

Inside `compiler/`, install `requirements-test.txt` in a dedicated environment,
then run `make test PYTHON=python`. `tests/test_ort_export.py` calls ONNX Runtime
1.20.1 `quantize_static` with QDQ format, QInt8 activation/weight types,
`per_channel=False`, and `ActivationSymmetric=True`, `WeightSymmetric=True`.
It compiles that exported model without a calibration file, verifies the target
binary against the integer oracle, and compares against ONNX Runtime execution.
Without the optional dependency this one test is reported as skipped.
