#!/usr/bin/env python3
"""Train a real 4->100->100->1 network and export compiler inputs.

From the repository root:
    python3 test-dnn-large/train.py
Dependencies: numpy, scikit-learn, torch, onnx.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


class Net41001001(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # 1. Real measurements; fit preprocessing ONLY on the training split.
    data = load_breast_cancer()
    features, gain = [0, 2, 3, 7], 25.0
    train_raw, test_raw, train_y, test_y = train_test_split(
        data.data[:, features], data.target, test_size=0.2,
        random_state=args.seed, stratify=data.target)
    scaler = StandardScaler().fit(train_raw)
    train_x = (scaler.transform(train_raw) * gain).astype(np.float32)
    test_x = (scaler.transform(test_raw) * gain).astype(np.float32)

    # 2. Two 100-neuron ReLU layers and one binary-classification logit.
    model = Net41001001()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = criterion(model(torch.from_numpy(train_x)), torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={loss.item():.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        float_outputs = model(torch.from_numpy(test_x)).numpy()

    # 3. Export dense/ReLU layers with a variable batch size.
    model_path = out / "model.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 4), str(model_path), opset_version=17,
        input_names=["inputs"], output_names=["logits"],
        dynamic_axes={"inputs": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False)
    exported = onnx.load(model_path)
    onnx.checker.check_model(exported)
    # Print every initializer without truncation; nine significant digits
    # preserve each trained FLOAT32 value when read back as FLOAT32.
    with (out / "model_values.log").open("w") as log:
        for tensor in exported.graph.initializer:
            values = onnx.numpy_helper.to_array(tensor)
            log.write(f"{tensor.name}: shape={values.shape}, dtype={values.dtype}\n")
            log.write(np.array2string(values, threshold=values.size, precision=9,
                                     floatmode="unique") + "\n\n")
    onnx_outputs = ReferenceEvaluator(exported).run(None, {"inputs": test_x})[0]
    np.testing.assert_allclose(onnx_outputs, float_outputs, rtol=1e-5, atol=1e-5)

    # 4. Save only compiler inputs, held-out inputs, and preprocessing metadata.
    for name, values in {
        "model.calibration": train_x, "inputs": test_x,
    }.items():
        np.save(out / f"{name}.npy", values)
    save_json(out / "preprocessing.json", {
        "feature_indices": features,
        "feature_names": [str(data.feature_names[i]) for i in features],
        "standard_scaler_mean": scaler.mean_.tolist(),
        "standard_scaler_scale": scaler.scale_.tolist(), "feature_gain": gain,
        "input_formula": "((raw_features - mean) / scale) * feature_gain",
        "class_names": data.target_names.tolist(),
        "output_note": "One logit for class 1 (benign): logit >= 0 predicts benign, otherwise malignant. Sigmoid is host-side only."})

    accuracy = float(np.mean((float_outputs[:, 0] >= 0).astype(np.int64) == test_y))
    print(f"Float test accuracy: {accuracy:.2%}")
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
