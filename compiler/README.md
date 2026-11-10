# Tiny NPU Compiler — v0.3.0

Compile a trained **float or quantized dense/ReLU model** into the Tiny NPU's 128-bit instruction
format, inspect every instruction, and execute the emitted SRAM image with a
functional interpreter.

This package targets the current architecture: one 1024×128-bit SRAM, a
programmable controller, two operand slots **inside the MAC engine**, a 4×4
outer-product array, separate activation and scale/clamp blocks, and stream
input/output endpoints. It does not add a SIMD engine or an SoC slave interface.

**Integration status:** this is the project's host-side compiler and instruction-set
interpreter. It does not modify the previous C model. That model must implement
the ISA v1 contract in `docs/ISA.md` before it can execute these binaries.
The outer instruction fields and opcode numbers follow the latest diagram.
Previously unspecified inner fields are now defined explicitly as ISA v1.

## 1. Start here

This directory belongs at `Tiny-NPU/compiler/`. This release contains only that
directory; it does not change the C model, architecture drawings, or root build.
The standalone Linux x86-64 executable is **`compiler/bin/tiny-npu-compiler`**.
From your `Tiny-NPU/` root:

```sh
chmod +x compiler/bin/tiny-npu-compiler
# Compatible quantized ONNX from PTQ or QAT: embedded scales, no calibration file.
./compiler/bin/tiny-npu-compiler model_qdq.onnx
# Float ONNX: representative input samples are needed for PTQ.
./compiler/bin/tiny-npu-compiler model.onnx --calibration calibration.npy
```

Output defaults to `MODEL_STEM_compiled/` beside the input model. See
[BINARY.md](BINARY.md) for executable usage and
[docs/QUANTIZED_IMPORT.md](docs/QUANTIZED_IMPORT.md) for the supported export contract.
PTQ and QAT describe how a model was prepared; the compiler reads its exported
operators, integer values, scales and zero points. It does not retrain a model.

| Input | Compiler action | Calibration `.npy` |
|---|---|---|
| Float ONNX / float JSON | Calibrate, then quantize | Required |
| Compatible INT8-weight Q/DQ ONNX | Preserve integer weights and embedded scales | Not needed |
| Compatible float-weight Q/DQ ONNX | Evaluate constant weight Q nodes; preserve scales | Not needed |
| Per-channel, asymmetric, UINT8, dynamic or QOperator graph | Reject with diagnostic | Does not make it supported |

The target supports **signed INT8, zero point 0, scalar per-tensor scales**.
Existing scales are never silently recalibrated. Passing `--calibration` for a
quantized model is an error. A calibration sidecar is ignored for that model.

### Run from Python source instead

Use Python 3.10 or newer. From this `compiler/` directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make demo PYTHON=python
make disasm PYTHON=python
make test PYTHON=python
```

On Windows, activate `.venv\Scripts\activate` and use the equivalent Python
commands below if Make is unavailable:

```sh
python examples/make_example.py
python -m tiny_npu compile --model examples/model.onnx --calibration examples/calibration.npy --batch 4 --out build/example
python -m tiny_npu disasm build/example/program.bin
python -m tiny_npu run --image build/example --inputs examples/inputs.npy --out build/example/run.json
python -m unittest discover -s tests -v
```

The ZIP already contains `examples/` input files and a **compiled_example/**
directory, plus `compiled_qdq_example/`, so you can inspect `program.asm`, `program.hex`, and `docs/ISA.md`
before installing anything. The example is a deterministic, untrained 4→4→4
network with seed 42. It is **not your real trained model**.

## 2. Compile your own trained model

### PyTorch

The compiler accepts ONNX, not a standalone `.pth` state dictionary. Export
your exact training architecture in evaluation mode. The optional
`examples/export_pytorch.py` template matches your `Net444` class:

```sh
python examples/export_pytorch.py trained_state_dict.pth model.onnx
```

Edit that template if layer names, dimensions, or activation functions differ.
It uses the legacy exporter (`dynamo=False`) explicitly to produce a simple
opset-17 dense graph. PyTorch's modern exporter is also usable if its resulting
graph is within the supported subset. PyTorch is an optional export dependency,
not installed by this compiler's requirements file.

### TensorFlow / Keras

Export a TensorFlow SavedModel using your training environment, then convert it
with tf2onnx in that environment:

```sh
python -m tf2onnx.convert --saved-model saved_model_dir --opset 17 --output model.onnx
```

TensorFlow/tf2onnx dependencies belong to the export environment. Their versions
must be compatible with one another. Neither TensorFlow nor PyTorch export was
executed as part of this package's validation. The ONNX frontend itself was
tested against ONNX's reference evaluator.

### Calibration and compilation

Save representative **preprocessed floating-point model inputs** into
`calibration.npy`, shape `[number_of_samples, number_of_features]`. For example,
in your training script:

```python
import numpy as np
np.save("calibration.npy", np.asarray(X_train, dtype=np.float32))
np.save("inputs.npy", np.asarray([[1, 2, 3, 4]], dtype=np.float32))
```

This calibration path applies to float models only. For compatible quantized
ONNX, compile without `--calibration`. Use the same feature normalization as training. Labels are not needed for range
calibration. Use a separate evaluation set with labels to measure model accuracy.

```sh
python -m tiny_npu compile --model model.onnx --calibration calibration.npy --batch 4 --out build/my_model
python -m tiny_npu disasm build/my_model/program.bin
python -m tiny_npu run --image build/my_model --inputs inputs.npy --out build/my_model/run.json
```

Use `--batch 1`, `2`, `3`, or `4` at compile time. The runner splits input samples
into compiled-size batches and zero-pads the final incomplete batch. The extra
vectors do not affect real vectors. Only real outputs are returned.

By default, runtime inputs are **float features** and are quantized using the
compiled input scale. Add `--quantized` only when the input file already contains
raw INT8 codes. Supplying `[1,2,3,4]` in these two modes can produce different
results because the interpretations differ.

## 3. Compiler flow and files

| Stage | Code | Input | Output |
|---|---|---|---|
| Parse and normalize | `frontend.py`, `ir.py` | `.onnx` and any adjacent external tensor files | Dense layers with `W[neuron,input]` |
| Import embedded quantization | `qdq.py` | Q/DQ ONNX with static parameters | Integer layers and import report; no calibration |
| Calibrate and quantize | `quantize.py` | Float layers + `calibration.npy` | INT8 W, INT32 bias, per-layer scales and M/S |
| Tile, allocate, schedule | `compiler.py` | Quantized layers, batch size | 4×4 operand tiles, addresses, instructions |
| Encode | `isa.py` | Typed instructions | Little-endian 128-bit words |
| Emit image and reports | `compiler.py` | Program and parameter allocations | Files below |
| Inspect | `isa.py`, CLI `disasm` | `program.bin` | Disassembly derived from binary bytes |
| Execute/check | `interpreter.py`, `reference.py`, CLI `run` | `sram.bin`, input vectors, compiler metadata | Per-k trace and exact integer comparisons |

| Generated file | Meaning |
|---|---|
| `program.bin` | Only instructions, 16 bytes/instruction, little-endian |
| `program.hex` | One instruction word `[127:0]` per text line |
| `program.asm` | Byte PC, hex word, decoded operation and operands |
| `params.bin` | Address-preserving image for `[0x0000,0x3900)`, including zero activation gaps |
| `sram.bin` | Complete 16 KiB initialization image including program |
| `sram.hex` | 1024 lines, each a 128-bit word, suitable for a 128-bit `$readmemh` array |
| `memory_map.txt` | Every allocation, byte address, size, region |
| `manifest.json` | ISA version, entry PC, dimensions, scales, packing, addresses, checksums |
| `import_report.json` | Import mode, original scales, bias conversion and M/S approximation error |
| `quantization.json` | Exact integer weights/biases and scale configuration |
| `schedule.json` | Instruction annotations and tile indices, **not clock cycles** |
| `run.json` | Generated by `run`, contains per-k accumulators and outputs |

Stages share in-memory Python objects. Intermediate `.json` files are emitted
for inspection; the interpreter executes **the SRAM binary bytes**, not those
JSON instructions. The checking layer uses JSON only for the independent oracle
and for mapping trace entries to logical layers.

## 4. Numerics

Symmetric per-tensor quantization uses zero point 0. For **float-model PTQ**, each tensor scale is its
calibration maximum absolute value divided by 127; an all-zero tensor uses 1.
Runtime quantization rounds to nearest, ties to even, and clamps to [-128,127].
All calculations used to choose scales are in float64 after importing model
parameters. This is deterministic but may differ from another tool's float32 PTQ.

For imported Q/DQ, the source scales and existing INT8 weight codes are preserved.
FLOAT Q/DQ weight constants are evaluated with FLOAT32 division and ties-to-even
rounding. Compatible INT32 biases are preserved; float biases are converted to
accumulator units. See `import_report.json` for conversion error.

For each layer:

```text
Wq = clamp_int8(round_even(Wfloat / weight_scale))
Bq = round_even(Bfloat / (input_scale * weight_scale))
acc[neuron,vector] = Bq[neuron] + sum_k(Wq[neuron,k] * Xq[vector,k])
activated = ReLU(acc) or acc
Yq = clamp_int8(round_even(activated * M / 2**S))
M / 2**S approximates input_scale * weight_scale / output_scale
```

Bias enters once, when accumulation starts. Subsequent reduction tiles retain
the accumulator. Post-processing happens only after all reduction tiles.
M is positive signed-INT32, S is 0..62, and multiplication uses signed INT64.
The compiler bounds every possible INT8 input's partial accumulation and rejects
possible INT32 overflow. Runtime interpreter overflow checking is also enabled.

The final output scale is **fixed at compile time**. It is not recomputed from
each inference. Your previous Python function's per-inference output scaling
will therefore produce different INT8 display values. Use `run.json`'s
`output_scale` to interpret the compiled output.

Integer bit-exactness is checked between the emitted program and an independent
oracle. It does not mean float and quantized predictions are identical, or that
your application's accuracy is preserved. Evaluate that separately.

## 5. Current capabilities and boundaries

- Sequential dense networks, rank-2 input `[batch,features]`, constant weights
  and bias, ReLU or no activation. Supports `Gemm`, `MatMul` + optional constant
  `Add`, runtime `Identity`, tensor `Constant`, and constant `Transpose`/`Identity`.
- Float ONNX opsets 7..21; quantized Q/DQ opsets 13..21. No convolution,
  residuals, dynamic weights, Softmax, Sigmoid, LeakyReLU, QOperator kernels,
  per-channel/asymmetric quantization, or arbitrary graph optimizations.
  Unsupported graphs fail explicitly. See the quantized-import guide for details.
- Arbitrary positive dense dimensions within fixed SRAM and program capacity.
  Neuron and reduction dimensions split into tiles of at most 4. Non-multiples
  of 4 are zero-padded, with explicit active counts in each MAC instruction.
- Two internal operand slots are used. Scheduling is conservative and
  dependency-ordered. There is no latency-aware prefetch optimizer yet.
- Hidden forwarding is enabled for complete hidden tensors of at most four
  channels. Larger hidden tensors use SRAM. If further neuron tiles also need
  that hidden tensor, its SRAM copy is retained. `--no-forward` disables bypass.
- Instructions are unrolled for transparent disassembly. Loop encoding and
  interpreter support are included, but automatic loop compression is not.
- This is a **functional** instruction interpreter. It does not model bus
  back-pressure, concurrent engine execution, Fetch/Decode clock counts, or
  cycles/stalls. `useful_macs` counts arithmetic work only. A future RTL/C decoder
  must enforce slot/accumulator/config dependencies and single-port arbitration.
- Compilation fails if any region exceeds capacity, including the **112-word
  instruction limit**. It does not silently spill data into another region.

## 6. Reading the package

Start with `docs/BINARY_WALKTHROUGH.md`, then `docs/ISA.md`. In the source, read
`ir.py` → `frontend.py` / `qdq.py` → `quantize.py` → `compiler.py` → `isa.py`.
`interpreter.py` explains execution. `reference.py` is independent of ISA and
SRAM layout, and uses rational-number rounding for numerical cross-checks.

`make test` covers encoding, signed rounding, ONNX import, all batch sizes,
forwarding on/off, multi-tile and tail dimensions, partial runtime batches,
per-k accumulator equality, post outputs, final results, capacity rejection,
invalid instructions/configuration/operands, FIFO misuse and INT32 overflow.

## 7. Source references for frontend/export semantics

- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [ONNX DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html)
- [ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)
- [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html)
- [tf2onnx conversion](https://github.com/onnx/tensorflow-onnx)

These describe input/export formats. Tiny NPU ISA v1 is this project's own ISA;
the resulting binary is not executable on a Coral NPU or a general CPU.
