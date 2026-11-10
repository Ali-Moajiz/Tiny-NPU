"""Create a reproducible toy model; these are NOT the user's trained weights."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    target = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 0.7, (4, 4)).astype(np.float32)
    w2 = rng.normal(0, 0.7, (4, 4)).astype(np.float32)
    b1 = rng.normal(0, 0.2, 4).astype(np.float32)
    b2 = rng.normal(0, 0.2, 4).astype(np.float32)
    layers = [dict(name="fc1", weights=w1.tolist(), bias=b1.tolist(), activation="relu"),
              dict(name="fc2", weights=w2.tolist(), bias=b2.tolist(), activation="none")]
    (target / "model.json").write_text(json.dumps(dict(format="tiny-npu-float-v1", layers=layers), indent=2)+"\n")
    nodes = [helper.make_node("Gemm", ["input", "W1", "B1"], ["z1"], name="fc1", transB=1),
             helper.make_node("Relu", ["z1"], ["h1"], name="relu"),
             helper.make_node("Gemm", ["h1", "W2", "B2"], ["output"], name="fc2", transB=1)]
    tensors = [numpy_helper.from_array(x, name) for x, name in ((w1,"W1"),(b1,"B1"),(w2,"W2"),(b2,"B2"))]
    graph = helper.make_graph(nodes, "example444",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, target / "model.onnx")
    np.save(target / "calibration.npy", rng.uniform(-4, 4, (256, 4)).astype(np.float32))
    np.save(target / "inputs.npy", np.array([[1,2,3,4], [0,1,-4,-1], [0,0,0,0], [-1,2,-3,4]], dtype=np.float32))
    print(f"Example files written to {target}")


if __name__ == "__main__":
    main()
