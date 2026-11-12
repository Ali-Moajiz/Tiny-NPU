Second model: **4 → 100 → 100 → 1**, with ReLU after each hidden layer.
The original `test-dnn/` model is unchanged.

Run from the repository root:

```bash
python3 test-dnn-large/train.py
```

Dependencies: `torch`, `scikit-learn`, `numpy`, and `onnx`.
Uses the real breast-cancer dataset bundled with scikit-learn, four selected
features, a stratified 80/20 split, and preprocessing fitted on training data only.
Training defaults to 300 epochs with seed 42; override with `--epochs` and `--seed`.

Training writes only these files in `test-dnn-large/artifacts/`:

- `model.onnx`: trained graph, weights, and biases.
- `model.calibration.npy`: preprocessed training inputs.
- `inputs.npy`: held-out preprocessed test inputs.
- `preprocessing.json`: feature order, normalization, and output interpretation.
- `model_values.log`: all trained weights and biases.

The output is one raw logit, trained using `BCEWithLogitsLoss`.
A logit >= 0 predicts class 1 (benign); a negative logit predicts class 0
(malignant). Apply sigmoid on the host if a probability is needed; sigmoid is
not included in ONNX so that the graph uses only dense and ReLU operations.
The original model's argmax-based inference is not suitable for this single output.

Training does not compile or run inference. Compile separately:

```bash
python3 compiler/main.py test-dnn-large/artifacts/model.onnx \
  --calibration test-dnn-large/artifacts/model.calibration.npy \
  --out test-dnn-large/compiled --debug
```

With the updated 1 MiB SRAM map, this model fits: its padded INT8 weights
require 10,800 bytes of the 768 KiB weight region, and its 2,181 instructions
fit the 3,840-instruction program region. Use the updated compiler executable
and C model together; old 16 KiB builds cannot accommodate this model.

After changing the architecture, rerun training to replace any previously
exported ONNX model before compiling.
