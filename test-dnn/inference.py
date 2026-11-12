#!/usr/bin/env python3
"""Float ONNX and independent INT8 inference; no compiler or PyTorch imports.

python3 test-dnn/inference.py
python3 test-dnn/inference.py --inputs npu_inputs.npy --input-mode int8 --compare npu_outputs.npy
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def integer_inference(inputs, layers):
    """Exact Python integer MACs and rational ties-to-even rounding.

    Uses the compiler's exported parameters, but executes no NPU instructions
    and calls none of the compiler's arithmetic/reference functions.
    """
    x = np.asarray(inputs).tolist()
    outputs = []
    for layer in layers:
        next_x = []
        for sample in x:
            row = []
            for weights, bias in zip(layer["weights"], layer["bias"]):
                acc = int(bias)
                for value, weight in zip(sample, weights):
                    acc += int(value) * int(weight)
                    if not -(1 << 31) <= acc < (1 << 31):
                        raise OverflowError("INT32 accumulator overflow")
                if layer["activation"] == "relu":
                    acc = max(0, acc)
                scaled = round(Fraction(acc * int(layer["multiplier"]),
                                        1 << int(layer["shift"])))
                row.append(max(-128, min(127, scaled)))
            next_x.append(row)
        x = next_x
        outputs.append(x)
    return np.asarray(x, dtype=np.int8), outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--quantization", type=Path,
                        default=Path(__file__).resolve().parent / "compiled" / "quantization.json",
                        help="Integer parameters from a matching compiler --debug build")
    parser.add_argument("--inputs", type=Path, help="NPY shaped [samples,4]; defaults to artifacts/inputs.npy")
    parser.add_argument("--input-mode", choices=["model", "features", "int8"], default="model",
                        help="Preprocessed float, original four features, or raw signed INT8 codes")
    parser.add_argument("--compare", type=Path, help="NPU output NPY [samples,2], raw INT8 codes; exact comparison")
    parser.add_argument("--out", type=Path, help="Save full results to JSON only when requested")
    args = parser.parse_args()
    root = args.artifacts
    if not args.quantization.is_file():
        parser.error("Compile with --debug first; see test-dnn/README.md for the command")
    layers = json.loads(args.quantization.read_text())
    metadata = json.loads((root / "preprocessing.json").read_text())
    x = np.load(args.inputs or root / "inputs.npy", allow_pickle=False)
    width = len(layers[0]["weights"][0])
    if x.ndim != 2 or not len(x) or x.shape[1] != width or not np.isfinite(x).all():
        parser.error(f"Inputs must be finite, nonempty [samples,{width}]")
    input_scale = layers[0]["input_scale"]
    if args.input_mode == "int8":
        if np.any(x != np.rint(x)) or np.any(x < -128) or np.any(x > 127):
            parser.error("INT8 inputs must contain integers in [-128,127]")
        xq = x.astype(np.int8)
        model_x = xq.astype(np.float64) * input_scale
    else:
        model_x = np.asarray(x, dtype=np.float64)
        if args.input_mode == "features":
            model_x = ((model_x - metadata["standard_scaler_mean"])
                       / metadata["standard_scaler_scale"] * metadata["feature_gain"])
        xq = np.clip(np.rint(model_x / input_scale), -128, 127).astype(np.int8)
    float_output = ReferenceEvaluator(onnx.load(root / "model.onnx")).run(
        None, {"inputs": model_x.astype(np.float32)})[0]
    output, layer_outputs = integer_inference(xq, layers)
    result = {
        "input_mode": args.input_mode, "input_int8": xq.tolist(),
        "output_int8": output.tolist(), "layer_outputs_int8": layer_outputs,
        "output_scale": layers[-1]["output_scale"],
        "output_dequantized": (output.astype(np.float64) * layers[-1]["output_scale"]).tolist(),
        "output_float": float_output.tolist(),
        "predictions_int8": output.argmax(1).tolist(),
        "predictions_float": float_output.argmax(1).tolist(),
        "class_names": metadata["class_names"],
    }
    if args.compare:
        actual = np.load(args.compare, allow_pickle=False)
        if actual.shape != output.shape:
            parser.error(f"NPU output shape {actual.shape} differs from expected {output.shape}")
        mismatches = np.argwhere(actual != output)
        if len(mismatches):
            sample, channel = map(int, mismatches[0])
            parser.exit(1, f"FAIL: {len(mismatches)} mismatches; first at [{sample},{channel}]: "
                           f"NPU={actual[sample, channel]}, expected={output[sample, channel]}\n")
        result["comparison_passed"] = True
        print(f"PASS: all {len(output)} samples exactly match independent INT8 inference")
    print(f"First INT8 input: {xq[0].tolist()} -> output: {output[0].tolist()}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Inference results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
