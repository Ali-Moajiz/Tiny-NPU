# Tiny-NPU

Tiny-NPU runs small, sequential dense neural networks using a 4×4 INT8 MAC
array, INT32 accumulators, and one 1 MiB SRAM. The Python compiler converts a
trained ONNX model into instructions and memory images. The C model executes
those images and reports outputs, accumulator activity, and simulated cycles.

This guide takes you from a Python model to execution in the **C simulator**.
Commands below run from the `Tiny-NPU/` repository root.

```text
Train in Python → export ONNX + calibration inputs → compile → run C model
```

## NPU specifications

| Component | Specification |
|---|---|
| Compute array | 4 × 4 multiply-accumulate units |
| Inputs and weights | Signed INT8 |
| Biases and accumulators | Signed INT32 |
| Post-processing | Optional ReLU, integer scaling, nearest-even rounding, INT8 saturation |
| Operand storage | Two operand slots |
| SRAM | One 1 MiB block, 65,536 × 128-bit words |
| Instruction width | 128 bits |
| Addressing | 32-bit byte addresses; 16-byte aligned transfers |
| Batch size | 1–4 input vectors per execution |
| Program capacity | 3,840 instructions |
| Host stream capacity | 256 beats each for input and output; up to 1,024 features |

The C model executes instructions sequentially and models registered SRAM
reads, stream back-pressure, and selectable post-processing latency. It reports
simulated cycles rather than measurements from a physical chip.

## Run your model

### 1. Prepare dependencies

You need Python 3.10–3.12, a C99 compiler, and Make. For Python compilation and verification, install
NumPy and ONNX using the compiler requirements below. A virtual environment is recommended:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r compiler/requirements.txt
```

If virtual-environment creation reports that `ensurepip` is unavailable on
Ubuntu/Debian, install the matching `python3-venv` system package first.

The included Linux x86-64 compiler executable can compile models without
Python dependencies. Training and Python inference still need their dependencies.

### 2. Prepare your trained model

Provide a trained ONNX model that meets the [supported DNN requirements](#supported-dnns).
For a floating-point model, prepare:

| File | What it contains |
|---|---|
| `my-model/model.onnx` | Network graph and trained weights and biases |
| `my-model/model.calibration.npy` | Representative preprocessed float inputs, shaped `[samples, input_features]` |
| `my-model/inputs.npy` | Inputs for verification, with the same preprocessing |

Calibration inputs do not need labels. Use representative training inputs or
an independent calibration set; reserve held-out data for accuracy evaluation.
Keep your feature order and preprocessing parameters with your application.

### 3. Compile

```sh
chmod +x compiler/bin/tiny-npu-compiler
./compiler/bin/tiny-npu-compiler my-model/model.onnx \
  --calibration my-model/model.calibration.npy \
  --out compiled --batch 4 --debug
```

Alternatively, use `python3 compiler/main.py` instead of the executable.
The `--calibration` argument can point to any suitable NPY file. Without it,
the compiler searches for `model.calibration.npy`, then `calibration.npy`, beside
the floating-point ONNX model.

The compiler reads network structure and parameters from ONNX, runs calibration
inputs through the float model to choose scales, and generates INT8 weights,
INT32 biases, and integer scaling settings. It does not train the model.

| Compiled file | Purpose |
|---|---|
| `program.bin` | Encoded 128-bit instructions only |
| `params.bin` | 987,136-byte memory prefix before the program, including gaps |
| `sram.bin` | Complete 1,048,576-byte SRAM image |
| `sram.hex` | Complete SRAM image for the C model: 65,536 words |
| `program.asm`, `program.hex` | Readable assembly and instruction-only HEX |
| `manifest.json` | Batch, dimensions, scales, addresses, and allocations |
| `quantization.json` | Exact integer parameters used by the NPU |

HEX, assembly, and JSON reports require `--debug`. The C model needs `sram.hex`.

### 4. Run the C model

Set the feature counts below to your network's input and output widths.
The values 8 and 2 are illustrative dimensions, not fixed NPU requirements.
Run 100 random INT8 batches with a repeatable seed:

```sh
make -C c-model run TB=1 BATCH=4 LOG=flow SRAM=../compiled/sram.hex \
  ARGS="--runs 100 --seed 42 --input-features 8 --output-features 2" \
  2>&1 | tee npu.log
```

Or enter your own quantized inputs, one vector per prompt:

```sh
make -C c-model run TB=1 BATCH=4 LOG=flow SRAM=../compiled/sram.hex \
  ARGS="--interactive --input-features 8 --output-features 2"
```

Enter one signed INT8 integer per input feature for each vector; type `q` to quit.
For the illustrative command above, that means eight values in `[-128, 127]`.
With batch 4, provide four vectors for each inference. `BATCH` must match
`--batch` used at compilation. Feature counts must match your model's input
and output widths.

`LOG=none` shows results/metrics without cycle traces; `flow`, `ports`, and
`both` add logging. `POST_LATENCY=0|1` selects the supported post-processing
timing. `--max-cycles N` overrides the default 100,000-cycle limit.
Random INT8 inputs exercise execution; they do not measure classification accuracy.

### 5. Verify with your real inputs

Generate expected integer results for the held-out inputs:

```sh
./compiler/bin/tiny-npu-compiler run --image compiled \
  --inputs my-model/inputs.npy \
  --out compiled/reference.json
```

This checks binary execution against an independent integer reference. It
records quantized inputs, integer outputs, dequantized outputs, and traces.
To execute the **first batch of those exact inputs** in the C model, pack them:

```sh
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("compiled/manifest.json").read_text())
reference = json.loads(Path("compiled/reference.json").read_text())
rows = reference["input_int8"][:manifest["batch"]]
features = manifest["widths"][0]
packed = bytearray()
for start in range(0, features, 4):
    beat = bytearray(16)
    for vector, row in enumerate(rows):
        for channel in range(min(4, features - start)):
            beat[4 * vector + channel] = row[start + channel] & 255
    packed.extend(beat)
Path("input.bin").write_bytes(packed)
PY

make -C c-model run TB=0 BATCH=4 LOG=none SRAM=../compiled/sram.hex \
  ARGS="--input-bin ../input.bin --output-bin ../output.bin"
```

Unpack and compare every output code in that batch:

```sh
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

manifest = json.loads(Path("compiled/manifest.json").read_text())
reference = json.loads(Path("compiled/reference.json").read_text())
batch, features = manifest["batch"], manifest["widths"][-1]
data = np.frombuffer(Path("output.bin").read_bytes(), dtype=np.int8)
assert data.size == 16 * ((features + 3) // 4), "Unexpected output length"
tiles = data.reshape(-1, 4, 4)
actual = np.concatenate(list(tiles), axis=1)[:batch, :features]
expected = np.asarray(reference["output_int8"][:batch], dtype=np.int8)
np.testing.assert_array_equal(actual[:len(expected)], expected)
print("PASS: C model output matches the integer reference")
PY
```

## Supported DNNs

Tiny-NPU processes **feed-forward, fully connected DNNs (multilayer perceptrons)**
with sequential dense layers and optional ReLU activations. Models must have
one rank-2 input and output (`[samples, features]`), with constant trained
weights and biases. Layer widths may differ and need not be multiples of four.

### Supported operations and limits

- Supported: `Gemm` or `MatMul`, optional constant bias, ReLU or no activation.
- One runtime input and one output; no residual branches or parallel paths.
- Unsupported: convolutions, attention, sigmoid/softmax inside the graph,
  and other operators outside the supported dense/ReLU chain.
- Float ONNX opsets 7–21.
- Compatible static quantized Q/DQ exports use signed INT8, zero point 0,
  and per-tensor scales, with opsets 13–21. Omit `--calibration` for these.
  UINT8, asymmetric/per-channel, dynamic and QOperator quantization are unsupported.
- Batch size is 1–4. Host stream buffers hold 256 beats, up to 1,024
  input/output features. SRAM regions and the 3,840-instruction program limit
  must also fit; neuron count alone does not determine whether a model fits.

### Preprocessing and interpreting outputs

Use exactly the same feature order and preprocessing for calibration and real
inference. Fit any normalization on the training set, retain its parameters,
and apply it on the host before quantization. Keep preprocessing outside the
exported dense/ReLU graph.
Calibration NPY files contain representative inputs, not labels or weights.

For preprocessed floating-point input `x`, quantize on the host using the
compiled manifest's `input_scale`:

```python
x_q = np.clip(np.rint(x / input_scale), -128, 127).astype(np.int8)
```

The C model receives those signed INT8 codes. Its outputs are INT8 codes;
approximate float logits are `output_int8 * output_scale`, with `output_scale`
from the manifest. For binary classification trained with a single logit,
use the zero threshold; for multiple class logits, use argmax. For regression,
interpret dequantized outputs according to your training targets. Quantization can create ties or change predictions, so
measure your own held-out accuracy. Hardware correctness comparisons should
check integer outputs against the integer reference, rather than requiring
exact equality with the original float model.

## Memory map

Compiler and C model must use the same map. Addresses are byte addresses and
transfers are aligned to 16 bytes. Entry PC is `0xF1000`.

| Region | Start | End exclusive | Capacity |
|---|---|---|---|
| Weights | `0x00000` | `0xC0000` | 768 KiB |
| Activations | `0xC0000` | `0xE0000` | 128 KiB |
| Biases | `0xE0000` | `0xF0000` | 64 KiB |
| Configuration | `0xF0000` | `0xF1000` | 4 KiB |
| Instructions | `0xF1000` | `0x100000` | 60 KiB |

Regenerate images created for the former 16 KiB map. A capacity error means
the model's tiled allocations or unrolled program do not fit; reducing layer
width/depth can help. More SRAM does not add new supported operators.

## Rebuild and test

Rebuild the executable after editing compiler Python source:

```sh
python3 -m pip install pyinstaller==6.16.0 pyinstaller-hooks-contrib==2025.9
make -C compiler binary PYTHON=python3
```

The bundled executable does not pick up source changes automatically.

```sh
make -C compiler test PYTHON=python3
make -C c-model test
```

Further details: [compiler guide](compiler/README.md),
[C model guide](c-model/README.md).
