# Tiny-NPU Compiler

Compile dense neural networks into instructions and parameter images for the Tiny-NPU.

## Quick start

Run commands from the `compiler/` directory. The included executable supports Linux x86-64 and does not require Python.

```sh
chmod +x bin/tiny-npu-compiler

# Float ONNX model with representative calibration inputs.
./bin/tiny-npu-compiler model.onnx --calibration samples.npy

# Compatible quantized ONNX model with embedded quantization parameters.
./bin/tiny-npu-compiler quantized_model.onnx
```

Calibration data must be a NumPy array shaped `[samples, input_features]`, using the same preprocessing as the model.

For float models, omitting `--calibration` searches for `model.calibration.npy`, then `calibration.npy`, beside the model. Quantized models must omit this argument.

## Options

| Option | Description |
|---|---|
| `--out DIR` | Output directory; defaults to `MODEL_STEM_compiled/` beside the model |
| `--batch 1..4` | Vectors processed together; default: 4 |
| `--debug` | Generate assembly, HEX, memory-map and quantization reports |
| `--no-forward` | Disable direct forwarding of hidden activations |
| `--version` | Show the compiler version |

## Outputs

| File | Contents |
|---|---|
| `program.bin` | Encoded 128-bit NPU instructions |
| `params.bin` | 987,136-byte parameter-region image (964 KiB), including address gaps |
| `sram.bin` | Complete 1 MiB SRAM initialization image, including the program |

## SRAM layout

One 1 MiB SRAM contains 65,536 words of 128 bits. ISA addresses are 32-bit
**byte addresses**; each transfer is 16-byte aligned. Physical word index is
`address / 16`, using 16 bits (byte-address bits [19:4]).

| Region | Start | End exclusive | Capacity |
|---|---|---|---|
| Weights | `0x00000` | `0xC0000` | 768 KiB |
| Activations | `0xC0000` | `0xE0000` | 128 KiB |
| Biases | `0xE0000` | `0xF0000` | 64 KiB |
| Configuration | `0xF0000` | `0xF1000` | 4 KiB |
| Instructions | `0xF1000` | `0x100000` | 60 KiB / 3,840 instructions |

Entry/reset PC is `0xF1000`. `params.bin` covers bytes `[0, ENTRY_PC)`,
including gaps, rather than concatenating only used parameters.
With `--debug`, `sram.hex` has exactly 65,536 data lines of 32 hexadecimal
digits, with the lowest-address byte at the right. `program.hex` contains
only the emitted instructions. The compiler and C model must use this same
map. **Regenerate old 16 KiB images and rebuild old compiler executables.**

## Supported models

- Sequential dense networks using `Gemm` or `MatMul` with optional constant bias, and ReLU or no activation.
- One rank-2 input and output, with constant weights.
- Float ONNX opsets 7–21, with representative calibration data.
- Quantized ONNX Q/DQ opsets 13–21 from compatible PTQ or QAT exports: signed INT8, zero point 0, and per-tensor scales.

Convolutions, residual branches, per-channel or asymmetric quantization, UINT8, dynamic quantization and QOperator graphs are unsupported.

Models exceeding the fixed SRAM regions or 3,840-instruction program capacity are rejected.

## Inspect and verify

Compile with debug reports enabled:

```sh
./bin/tiny-npu-compiler model.onnx \
    --calibration samples.npy \
    --out output \
    --debug
```

Disassemble the program or verify execution:

```sh
./bin/tiny-npu-compiler disasm output/program.bin

./bin/tiny-npu-compiler run \
    --image output \
    --inputs inputs.npy \
    --out output/run.json
```

Inputs are floating-point features by default. Add `--quantized` when `inputs.npy` contains raw INT8 codes.

Verification checks the emitted binary against an integer reference. It does not measure model accuracy or hardware clock cycles.

Recompiling replaces generated files and clears previous debug reports and `run.json` in the output directory.

## Execute in the C model

From the repository root, recompile the ONNX model into a new image:

```sh
./compiler/bin/tiny-npu-compiler model.onnx --calibration samples.npy --out output --debug
make -C c-model run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--runs 100 --seed 42 --input-features 4 --output-features 2"
```

Set batch and feature counts to match your model. Inputs to the C model are
signed INT8 codes, not raw floating-point features. Use `--interactive` in ARGS
for terminal input, or TB=0 with `--input-bin` and `--output-bin` for packed beats.
The C host stream buffers still hold 256 beats (up to 1,024 features per tensor),
batch remains 1–4, and operators/forwarding rules are unchanged. More SRAM does
not remove these limits or the 3,840-instruction limit.

## Run from source

Python 3.10–3.12 is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python main.py model.onnx --calibration samples.npy
```

`main.py` provides the same commands as the executable. Its `compile_file()` function shows the compilation pipeline; `isa.py` defines the instruction encoding.

## Development

Run the tests:

```sh
make test PYTHON=python
```

The optional exporter integration test requires `onnxruntime==1.20.1`.

Package source changes into `bin/tiny-npu-compiler`:

```sh
python3 -m pip install pyinstaller==6.16.0 pyinstaller-hooks-contrib==2025.9
make binary PYTHON=python
```

Rebuild the executable after changing Python source.

Third-party notices are available in `THIRD_PARTY_LICENSES.txt`.
