Run these commands yourself from the repository root.

Training needs `torch`, `scikit-learn`, `numpy`, and `onnx`:

```bash
python3 test-dnn/train.py
```

This trains a 4 → 8 → 8 → 2 network on real breast-cancer measurements and
creates exactly five files in `test-dnn/artifacts/`:

| File | Purpose |
|---|---|
| `model.onnx` | Graph with trained weights and biases |
| `model.calibration.npy` | Preprocessed training inputs for calibration |
| `inputs.npy` | Held-out, preprocessed test inputs |
| `preprocessing.json` | Feature order, normalization, and class names |
| `model_values.log` | Every trained weight and bias, without truncation |

Training prints loss and float test accuracy. It does not compile or launch
inference. Rerunning replaces these five files.

Compile separately when ready:

```bash
python3 compiler/main.py test-dnn/artifacts/model.onnx \
  --calibration test-dnn/artifacts/model.calibration.npy \
  --out test-dnn/compiled --debug
```

Compiler binaries and debug reports stay in `compiled/`. Its
`quantization.json` contains the exact INT8 weights, INT32 biases, and scaling
settings needed by independent inference. Recompile after retraining.

Inference needs only `numpy` and `onnx`:

```bash
# Print an example result; no files are saved by default.
python3 test-dnn/inference.py

# Explicitly save all input codes and expected outputs.
python3 test-dnn/inference.py --out test-dnn/results.json

# Compare your NPU's results for the default held-out inputs.
python3 test-dnn/inference.py --compare npu_outputs.npy

# Supply your own raw INT8 inputs and matching NPU outputs.
python3 test-dnn/inference.py \
  --inputs npu_inputs.npy --input-mode int8 --compare npu_outputs.npy
```

Feed the same INT8 codes to the NPU and the reference. The optional results
JSON contains `input_int8`, `output_int8`, intermediate layer outputs, float
ONNX logits, dequantized outputs, and predictions. NPU output NPY arrays must
be shaped `[samples, 2]`, in sample order, with signed INT8 output codes.
Unpack stream beats and remove inactive lanes before comparison.

Input modes are `model` (preprocessed floats, default), `features` (four original
measurements in preprocessing.json order), and `int8` (raw signed codes).
In INT8 mode, float ONNX inference uses dequantized input codes. Output channels
are malignant then benign; argmax selects the predicted class.

Integer inference uses Python integers and exact rational ties-to-even rounding,
without compiler imports or NPU instruction interpretation. Comparison is exact
and exits with status 1 on mismatches. Float outputs can differ from quantized
NPU outputs; use `output_int8` for hardware comparisons.
